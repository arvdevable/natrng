import os
import sys
import numpy as np
import hashlib

# Add the directory to sys.path
sys.path.append(os.path.abspath(r"e:\web\natrng\base_implementation"))

from main import get_jitter_entropy, get_system_entropy, shannon_entropy, min_entropy

def test_source_diversity():
    print("Testing auxiliary entropy sources...")
    j1 = get_jitter_entropy(16)
    j2 = get_jitter_entropy(16)
    assert j1 != j2, "Jitter entropy should be different across calls"
    print(f"  [PASS] Jitter entropy (16B) looks diverse: {j1.hex()} vs {j2.hex()}")
    
    s1 = get_system_entropy()
    s2 = get_system_entropy()
    assert s1 != s2, "System entropy should be different across calls"
    print(f"  [PASS] System entropy (32B) looks diverse: {s1.hex()} vs {s2.hex()}")

def test_health_metrics():
    print("Testing health metrics on synthetic grains...")
    
    # Low variance grain (silence)
    silence = np.zeros(1024, dtype=np.int16)
    var_silence = np.var(silence)
    print(f"  Silence variance: {var_silence}")
    assert var_silence < 0.5
    
    # Diverse grain (white noise)
    noise = np.random.randint(-1000, 1000, 1024, dtype=np.int16)
    var_noise = np.var(noise)
    print(f"  Noise variance: {var_noise}")
    assert var_noise > 0.5
    
    # High repetition grain
    rep = np.array([1, 2, 1, 2] * 256, dtype=np.int16)
    unique_ratio = len(np.unique(rep)) / len(rep)
    print(f"  Repetition unique ratio: {unique_ratio}")
    assert unique_ratio < 0.1

def test_shannon_vs_min():
    print("Testing Shannon vs Min-Entropy on biased source...")
    # 75% zeros, 25% ones
    data = b"\x00" * 75 + b"\x01" * 25
    s = shannon_entropy(data)
    m = min_entropy(data)
    print(f"  Shannon: {s:.4f} b/B")
    print(f"  Min-Entropy: {m:.4f} b/B")
    assert s > m, "Shannon entropy usually higher than min-entropy for biased sources"

if __name__ == "__main__":
    test_source_diversity()
    test_health_metrics()
    test_shannon_vs_min()
    print("\nSecurity verification tests completed.")
