import os
import sys
import numpy as np
import wave
import struct

# Add the directory to sys.path
sys.path.append(os.path.abspath(r"e:\web\natrng\base_implementation"))

from main import lsb_bits_to_bytes, bit_balance, read_entropy_file, load_wav

def test_bit_balance_empty():
    print("Testing bit_balance with empty input...")
    try:
        result = bit_balance(b"")
        assert result == 0.0
        print("  [PASS] bit_balance(b'') returned 0.0")
    except Exception as e:
        print(f"  [FAIL] bit_balance(b'') raised {e}")

def test_lsb_padding():
    print("Testing lsb_bits_to_bytes padding...")
    # '1' -> 1 bit, should pad with 7 zeros -> '10000000' (128)
    res = lsb_bits_to_bytes("1")
    assert res == b"\x80"
    print("  [PASS] lsb_bits_to_bytes('1') returned b'\\x80'")

def test_read_entropy_file_validation():
    print("Testing read_entropy_file validation...")
    path = "test_entropy.bin"
    
    # Invalid magic
    with open(path, "wb") as f:
        f.write(b"BADMAGIC")
    try:
        read_entropy_file(path)
        print("  [FAIL] read_entropy_file accepted invalid magic")
    except ValueError as e:
        print(f"  [PASS] Caught expected ValueError for magic: {e}")

    # Truncated lsb_len
    with open(path, "wb") as f:
        f.write(b"ENTROPY1")
        f.write(b"\x00\x00\x00") # only 3 bytes instead of 8
    try:
        read_entropy_file(path)
        print("  [FAIL] read_entropy_file accepted truncated lsb_len")
    except EOFError as e:
        print(f"  [PASS] Caught expected EOFError for lsb_len: {e}")

    # Truncated lsb_data
    with open(path, "wb") as f:
        f.write(b"ENTROPY1")
        f.write((10).to_bytes(8, "big")) # lsb_len = 10
        f.write(b"123") # only 3 bytes
    try:
        read_entropy_file(path)
        print("  [FAIL] read_entropy_file accepted truncated lsb_data")
    except EOFError as e:
        print(f"  [PASS] Caught expected EOFError for lsb_data: {e}")

    if os.path.exists(path):
        os.remove(path)

def test_load_wav_validation():
    print("Testing load_wav validation...")
    path = "test.wav"
    
    # Create a dummy WAV with 24-bit samples (should fail because we expect 16-bit)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(3) # 24-bit
        wf.setframerate(44100)
        wf.writeframes(b"\x00" * 30)
    
    try:
        load_wav(path)
        print("  [FAIL] load_wav accepted 24-bit WAV")
    except ValueError as e:
        print(f"  [PASS] Caught expected ValueError for 24-bit WAV: {e}")

    # Create a dummy WAV with 2 channels (stereo)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(b"\x00" * 40)
    
    try:
        load_wav(path)
        print("  [FAIL] load_wav accepted stereo WAV")
    except ValueError as e:
        print(f"  [PASS] Caught expected ValueError for stereo WAV: {e}")

    if os.path.exists(path):
        os.remove(path)

if __name__ == "__main__":
    test_bit_balance_empty()
    test_lsb_padding()
    test_read_entropy_file_validation()
    test_load_wav_validation()
    print("\nAll verification tests completed.")
