import hashlib
import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from main import avalanche_test, serial_correlation, monobit_test, runs_test, rng_health_report

def test_avalanche():
    print("Testing Avalanche Effect (SHA-256)...")
    score = avalanche_test(lambda x: hashlib.sha256(x).digest(), trials=100)
    print(f"Avalanche Score: {score:.4f}")
    assert 0.45 <= score <= 0.55, f"Avalanche score {score} out of expected range"

def test_statistical_tests():
    print("\nTesting Statistical Tests with random data...")
    # Generate 1MB of random data
    data = os.urandom(1024 * 1024)
    
    corr = serial_correlation(data)
    mono = monobit_test(data)
    runs = runs_test(data)
    
    print(f"Serial Correlation: {corr:.4f}")
    print(f"Monobit Ratio: {mono:.4f}")
    print(f"Runs Test Score: {runs:.4f}")
    
    rng_health_report(data)
    
    assert abs(corr) < 0.05, f"Serial correlation {corr} too high"
    assert 0.45 <= mono <= 0.55, f"Monobit ratio {mono} out of expected range"
    assert 0.45 <= runs <= 0.55, f"Runs test score {runs} out of expected range"

if __name__ == "__main__":
    try:
        test_avalanche()
        test_statistical_tests()
        print("\n[SUCCESS] All basic health tests passed verification.")
    except Exception as e:
        print(f"\n[FAILURE] Verification failed: {e}")
        sys.exit(1)
