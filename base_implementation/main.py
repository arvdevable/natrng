"""
Audio Entropy Pipeline
======================
Steps:
  1. Capture raw PCM audio (16-bit, 44.1 kHz, mono)
  2. Convert samples to binary stream
  3. Slice into grains
  4. Extract entropy via LSB extraction and SHA-256 hashing

Dependencies:
    pip install sounddevice numpy
"""

import hashlib
import os
import struct
import time
from typing import Generator, Optional

import numpy as np

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SAMPLE_RATE    = 44_100          # Hz
DURATION       = 10              # seconds  (10–30)
GRAIN_SIZE     = 1024            # samples per grain  (try 2048 too)
LSB_BITS       = 2               # how many LSBs to extract per sample (1 or 2)
CHANNELS       = 1               # mono

# Security Settings
WARMUP_SECONDS   = 1             # seconds to discard from start of recording
VAR_THRESHOLD    = 0.5           # min variance per grain to accept as entropy
MIN_UNIQUE_RATIO = 0.1           # min unique samples ratio per grain
AUTOCORR_THRESHOLD = 1         # max abs(lag-1 autocorrelation) to accept


# ─────────────────────────────────────────────
# STEP 1 — CAPTURE RAW AUDIO
# ─────────────────────────────────────────────

def record_audio(duration: int = DURATION,
                 sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Record mono PCM audio from the default microphone.
    Returns a 1-D int16 NumPy array of raw samples.
    """
    try:
        import sounddevice as sd
    except ImportError:
        raise ImportError("Install sounddevice:  pip install sounddevice")

    print(f"[*] Recording {duration}s (+{WARMUP_SECONDS}s warmup) @ {sample_rate} Hz …")
    raw = sd.rec(
        frames=(duration + WARMUP_SECONDS) * sample_rate,
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype="int16",
        blocking=True,
    )
    sd.wait()
    all_samples = raw.flatten()
    # Trim warmup
    trim_idx = WARMUP_SECONDS * sample_rate
    samples = all_samples[trim_idx:]
    print(f"[+] Captured {len(samples):,} samples (discarded {trim_idx:,} warmup samples).")
    return samples


def load_wav(path: str) -> np.ndarray:
    """
    Alternate loader: read a WAV file instead of live recording.
    Returns int16 mono samples.
    """
    import wave
    with wave.open(path, "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError("Need 16-bit WAV")
        if wf.getnchannels() != 1:
            raise ValueError("Need mono WAV")
        raw_bytes = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw_bytes, dtype=np.int16)
    print(f"[+] Loaded {len(samples):,} samples from '{path}'.")
    return samples


# ─────────────────────────────────────────────
# STEP 2 — CONVERT TO BINARY STREAM
# ─────────────────────────────────────────────

def samples_to_bitstring(samples: np.ndarray) -> str:
    """
    Convert each int16 sample to its 16-bit binary representation
    and concatenate into one long bitstring.

    e.g.  [-1234, 5678, …]  →  '1111101100101110 0001011000010010 …'
    """
    # struct.pack with 'h' gives little-endian signed 16-bit
    parts = []
    for s in samples:
        # reinterpret as unsigned 16-bit for display
        u16 = int(np.int16(s)) & 0xFFFF
        parts.append(f"{u16:016b}")
    return "".join(parts)


def samples_to_bytes(samples: np.ndarray) -> bytes:
    """Pack int16 array to raw bytes (little-endian)."""
    return samples.astype("<i2").tobytes()


# ─────────────────────────────────────────────
# STEP 3 — GRAIN STRATEGY
# ─────────────────────────────────────────────

def make_grains(samples: np.ndarray,
                grain_size: int = GRAIN_SIZE) -> Generator[np.ndarray, None, None]:
    """
    Yield successive non-overlapping slices of `grain_size` samples.
    Any trailing samples that don't fill a full grain are discarded.
    """
    total   = len(samples)
    n_grains = total // grain_size
    print(f"[*] Slicing into {n_grains} grains × {grain_size} samples each.")
    for i in range(n_grains):
        yield samples[i * grain_size : (i + 1) * grain_size]


# ─────────────────────────────────────────────
# STEP 4A — OPTION A: LSB EXTRACTION
# ─────────────────────────────────────────────

def extract_lsb(grain: np.ndarray, n_bits: int = LSB_BITS) -> bytes:
    """
    Extract the lowest `n_bits` bits from each sample in the grain.
    Returns bytes containing the concatenated LSBs. (Vectorized)
    """
    mask = (1 << n_bits) - 1
    lsbs = grain.view(np.uint16) & mask
    
    # Pack into bytes. For performance, we skip the bitstring 
    # and use bit-shifting if n_bits is small.
    if n_bits == 1:
        return np.packbits(lsbs.astype(np.uint8)).tobytes()
    
    # Fallback for n_bits=2 or others (still much faster than the old loop)
    # as mostly vectorized except for the final pack
    bit_parts = [f"{val:0{n_bits}b}" for val in lsbs]
    return lsb_bits_to_bytes("".join(bit_parts))


def lsb_bits_to_bytes(bitstring: str) -> bytes:
    """
    Pack a bitstring into bytes. The input bitstring will be padded with 
    trailing '0' bits to the next byte boundary if its length is not a 
    multiple of 8. Uses: pad = (8 - len(bitstring) % 8) % 8.
    """
    # Pad to multiple of 8
    pad = (8 - len(bitstring) % 8) % 8
    bitstring = bitstring + "0" * pad
    out = bytearray()
    for i in range(0, len(bitstring), 8):
        out.append(int(bitstring[i:i+8], 2))
    return bytes(out)


# ─────────────────────────────────────────────
# STEP 4B — OPTION B: SHA-256 HASH PER GRAIN (WHITENING)
# ─────────────────────────────────────────────

def hash_grain(grain: np.ndarray) -> bytes:
    """
    SHA-256 of the raw bytes of one grain.
    Output: 32 bytes (256 bits) of whitened entropy per grain.

    Whitening removes statistical bias while preserving unpredictability.
    """
    raw = grain.astype("<i2").tobytes()
    return hashlib.sha256(raw).digest()


# ─────────────────────────────────────────────
# STEP 4C — AUXILIARY ENTROPY SOURCES
# ─────────────────────────────────────────────

def get_jitter_entropy(n_bytes: int = 16) -> bytes:
    """
    Measure CPU execution jitter.
    Collects high-precision timing deltas and whitens them via hashing.
    """
    deltas = bytearray()
    prev = time.perf_counter_ns()
    # Oversample to ensure enough entropy before hashing
    for _ in range(n_bytes * 4):
        cur = time.perf_counter_ns()
        deltas.append((cur - prev) & 0xFF)
        prev = cur
    # Hash the raw deltas to condense and whiten
    return hashlib.sha256(deltas).digest()[:n_bytes]


def get_system_entropy() -> bytes:
    """Gather high-resolution system state (timing, PID, etc)."""
    state = [
        time.perf_counter_ns().to_bytes(8, "big"),
        time.process_time_ns().to_bytes(8, "big"),
        struct.pack("I", os.getpid()),
    ]
    return hashlib.sha256(b"".join(state)).digest()


def autocorrelation_check(grain: np.ndarray, lag: int = 1) -> float:
    """
    Calculate lag-N autocorrelation. 
    Ideally near 0.0 for white noise.
    """
    if len(grain) <= lag:
        return 0.0
    # Pearson correlation coefficient
    corr = np.corrcoef(grain[:-lag], grain[lag:])[0, 1]
    return float(corr) if not np.isnan(corr) else 1.0


# ─────────────────────────────────────────────
# ENTROPY QUALITY METRICS
# ─────────────────────────────────────────────

def shannon_entropy(data: bytes) -> float:
    """
    Calculate Shannon entropy in bits per byte.
    Perfect randomness → 8.0 bits/byte.
    """
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs  = counts[counts > 0] / len(data)
    return float(-np.sum(probs * np.log2(probs)))


def bit_balance(data: bytes) -> float:
    """Fraction of 1-bits. Ideal random source → 0.50."""
    if not data:
        return 0.0
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return float(bits.mean())


def min_entropy(data: bytes) -> float:
    """
    Calculate min-entropy in bits per byte.
    H_min = -log2(max_probability)
    Perfect randomness → 8.0 bits/byte.
    """
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    max_p = np.max(counts) / len(data)
    return float(-np.log2(max_p))


# ─────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────

def run_pipeline(source: str = "mic",
                 wav_path: Optional[str] = None,
                 duration: int = DURATION,
                 grain_size: int = GRAIN_SIZE,
                 lsb_bits: int = LSB_BITS,
                 output_file: str = "entropy_output.bin") -> None:
    """
    Full pipeline:
      source = 'mic'  → live recording
      source = 'wav'  → load from wav_path
    """
    t0 = time.perf_counter()

    # ── Step 1: Capture ──────────────────────
    if source == "mic":
        samples = record_audio(duration)
    elif source == "wav" and wav_path:
        samples = load_wav(wav_path)
    else:
        raise ValueError("source must be 'mic' or 'wav' (with wav_path)")

    # ── Step 2: Show binary preview ──────────
    preview_n  = 4
    bitstring  = samples_to_bitstring(samples[:preview_n])
    print(f"\n[Step 2] Binary preview (first {preview_n} samples):")
    for i, chunk in enumerate([bitstring[j:j+16] for j in range(0, len(bitstring), 16)]):
        print(f"  sample {i:02d}: {chunk}")

    # ── Step 3 + 4: Grain → Entropy ──────────
    lsb_pool:      bytearray = bytearray()
    hash_pool:     bytearray = bytearray()
    grain_entropies: list[float] = []
    
    rejected_variance = 0
    rejected_unique   = 0
    rejected_autocorr = 0

    print(f"\n[Step 3+4] Processing grains …")
    for grain in make_grains(samples, grain_size):
        # ── Health Test 1: Variance ─────────────
        var = np.var(grain)
        if var < VAR_THRESHOLD:
            rejected_variance += 1
            continue
            
        # ── Health Test 2: Repetition / Uniqueness
        unique_ratio = len(np.unique(grain)) / len(grain)
        if unique_ratio < MIN_UNIQUE_RATIO:
            rejected_unique += 1
            continue
            
        # ── Health Test 3: Autocorrelation ─────
        ac = autocorrelation_check(grain, lag=1)
        if abs(ac) > AUTOCORR_THRESHOLD:
            rejected_autocorr += 1
            continue

        # Stats per grain (on raw PCM)
        g_bytes = grain.astype("<i2").tobytes()
        grain_entropies.append(shannon_entropy(g_bytes))

        # Entropy Sources
        lsb_bytes = extract_lsb(grain, lsb_bits)
        
        jitter_bytes = get_jitter_entropy(16)
        system_bytes = get_system_entropy()

        # Mixing (Option B - Whitening)
        # We mix audio bytes, jitter, and system state before hashing
        combined_source = lsb_bytes + jitter_bytes + system_bytes
        h = hashlib.sha256(combined_source).digest()
        
        lsb_pool.extend(lsb_bytes)
        hash_pool.extend(h)

    # ── Comparison & Metrics ──────────────────
    raw_bytes_out  = samples_to_bytes(samples)
    lsb_bytes_out  = bytes(lsb_pool)
    hash_bytes_out = bytes(hash_pool)

    # Per-grain stats
    g_mean = float(np.mean(grain_entropies)) if grain_entropies else 0.0
    g_std  = float(np.std(grain_entropies))  if grain_entropies else 0.0

    print(f"\n{'─'*60}")
    print(f"Health Tests: {rejected_variance} low-var, {rejected_unique} low-unique, {rejected_autocorr} high-autocorr grains rejected.")
    print(f"{'─'*60}")
    print(f"{'METRIC':<20} | {'RAW PCM':<10} | {'LSB EXT':<10} | {'MIXED/HASH':<10}")
    print(f"{'─'*60}")
    
    def print_metric(name, raw, lsb, hashed, fmt=".4f"):
        print(f"{name:<20} | {raw:{fmt}} | {lsb:{fmt}} | {hashed:{fmt}}")

    print_metric("Shannon (bits/B)", 
                 shannon_entropy(raw_bytes_out),
                 shannon_entropy(lsb_bytes_out),
                 shannon_entropy(hash_bytes_out))
    
    print_metric("Min-Entropy (bits/B)",
                 min_entropy(raw_bytes_out),
                 min_entropy(lsb_bytes_out),
                 min_entropy(hash_bytes_out))

    print_metric("Bit Balance (1s)",
                 bit_balance(raw_bytes_out),
                 bit_balance(lsb_bytes_out),
                 bit_balance(hash_bytes_out))
    
    print(f"{'─'*60}")
    print(f"Per-Grain Shannon Entropy (Raw):")
    print(f"  Mean: {g_mean:.4f} bits/byte")
    print(f"  StdDev: {g_std:.4f}")
    print(f"{'─'*60}")

    # ── Save ─────────────────────────────────
    with open(output_file, "wb") as f:
        # Write both pools sequentially with a simple 8-byte length prefix each
        lsb_len  = len(lsb_bytes_out).to_bytes(8, "big")
        hash_len = len(hash_bytes_out).to_bytes(8, "big")
        f.write(b"ENTROPY1")          # magic
        f.write(lsb_len)
        f.write(lsb_bytes_out)
        f.write(hash_len)
        f.write(hash_bytes_out)

    elapsed = time.perf_counter() - t0
    print(f"\n[+] Saved entropy to '{output_file}'  ({elapsed:.2f}s total)")


# ─────────────────────────────────────────────
# READER UTILITY
# ─────────────────────────────────────────────

def read_entropy_file(path: str) -> tuple[bytes, bytes]:
    """
    Read a file saved by run_pipeline().
    Returns (lsb_bytes, hash_bytes).
    """
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"ENTROPY1":
            raise ValueError(f"Not a valid entropy file (magic: {magic!r})")
        
        len_bytes = f.read(8)
        if len(len_bytes) < 8:
            raise EOFError("Truncated file while reading lsb_len")
        lsb_len = int.from_bytes(len_bytes, "big")
        if lsb_len < 0:
            raise ValueError(f"Invalid negative lsb_len: {lsb_len}")
            
        lsb_data = f.read(lsb_len)
        if len(lsb_data) < lsb_len:
            raise EOFError(f"Truncated file: expected {lsb_len} bytes for lsb_data, got {len(lsb_data)}")
            
        len_bytes = f.read(8)
        if len(len_bytes) < 8:
            raise EOFError("Truncated file while reading hash_len")
        hash_len = int.from_bytes(len_bytes, "big")
        if hash_len < 0:
            raise ValueError(f"Invalid negative hash_len: {hash_len}")
            
        hash_data = f.read(hash_len)
        if len(hash_data) < hash_len:
            raise EOFError(f"Truncated file: expected {hash_len} bytes for hash_data, got {len(hash_data)}")
            
    return lsb_data, hash_data


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audio Entropy Pipeline")
    parser.add_argument("--source",     default="mic",   choices=["mic", "wav"])
    parser.add_argument("--wav",        default=None,    help="Path to WAV file (when --source=wav)")
    parser.add_argument("--duration",   default=10,      type=int,   help="Recording seconds (10–30)")
    parser.add_argument("--grain",      default=1024,    type=int,   help="Samples per grain")
    parser.add_argument("--lsb-bits",   default=2,       type=int,   choices=[1, 2], help="LSBs per sample")
    parser.add_argument("--output",     default="entropy_output.bin")
    args = parser.parse_args()

    run_pipeline(
        source      = args.source,
        wav_path    = args.wav,
        duration    = args.duration,
        grain_size  = args.grain,
        lsb_bits    = args.lsb_bits,
        output_file = args.output,
    )