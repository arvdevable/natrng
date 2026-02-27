# notes
```
E:\web\natrng\base_implementation>python main.py --duration 20 --grain 2048 --lsb-bits 1
[*] Recording 20s (+1s warmup) @ 44100 Hz …
[+] Captured 882,000 samples (discarded 44,100 warmup samples).

[Step 2] Binary preview (first 4 samples):
  sample 00: 0000001001111000
  sample 01: 0000001010010000
  sample 02: 0000001011111100
  sample 03: 0000001101010110

[Step 3+4] Processing grains …
[*] Slicing into 430 grains × 2048 samples each.

────────────────────────────────────────────────────────────
Health Tests: 0 low-var, 0 low-unique, 430 high-autocorr grains rejected.
────────────────────────────────────────────────────────────
METRIC               | RAW PCM    | LSB EXT    | MIXED/HASH
────────────────────────────────────────────────────────────
Shannon (bits/B)     | 7.0898 | 0.0000 | 0.0000
Min-Entropy (bits/B) | 3.5500 | 0.0000 | 0.0000
Bit Balance (1s)     | 0.5074 | 0.0000 | 0.0000
────────────────────────────────────────────────────────────
Per-Grain Shannon Entropy (Raw):
  Mean: 0.0000 bits/byte
  StdDev: 0.0000
────────────────────────────────────────────────────────────

[+] Saved entropy to 'entropy_output.bin'  (21.79s total)

E:\web\natrng\base_implementation>
```

ini disebabkan oleh `high-autocorr`, penulis mengetes code python dengan suara penulis (et al.) bernyanyi. hasilnya? karena kesalahan dari pengadaan `high-autocorr` menolak semua 430 grain. commit 255c055, sha 255c05550532e266ed2ff7737b7c2f04660fe7e0. `AUTOCORR_THRESHOLD = 0.3` -> `AUTOCORR_THRESHOLD = 0.95`

==

```
E:\web\natrng\base_implementation>python main.py --duration 20 --grain 2048 --lsb-bits 1
[*] Recording 20s (+1s warmup) @ 44100 Hz …
[+] Captured 882,000 samples (discarded 44,100 warmup samples).

[Step 2] Binary preview (first 4 samples):
  sample 00: 1111111111010010
  sample 01: 1111111111100011
  sample 02: 1111111111001000
  sample 03: 1111111110010110

[Step 3+4] Processing grains …
[*] Slicing into 430 grains × 2048 samples each.

────────────────────────────────────────────────────────────
Health Tests: 0 low-var, 4 low-unique, 384 high-autocorr grains rejected.
────────────────────────────────────────────────────────────
METRIC               | RAW PCM    | LSB EXT    | MIXED/HASH
────────────────────────────────────────────────────────────
Shannon (bits/B)     | 6.6445 | 7.9797 | 7.8408
Min-Entropy (bits/B) | 2.9860 | 7.4150 | 6.5850
Bit Balance (1s)     | 0.5002 | 0.4990 | 0.4961
────────────────────────────────────────────────────────────
Per-Grain Shannon Entropy (Raw):
  Mean: 6.0020 bits/byte
  StdDev: 0.4090
────────────────────────────────────────────────────────────

[+] Saved entropy to 'entropy_output.bin'  (21.54s total)

E:\web\natrng\base_implementation>
```

149c522


===

```
E:\web\natrng\base_implementation>python main.py --duration 20 --grain 2048 --lsb-bits 1
[*] Recording 20s (+1s warmup) @ 44100 Hz …
[+] Captured 882,000 samples (discarded 44,100 warmup samples).

[Step 2] Binary preview (first 4 samples):
  sample 00: 1111111011000001
  sample 01: 1111111100000010
  sample 02: 1111111101101101
  sample 03: 0000000000000010

[Step 3+4] Processing grains …
[*] Slicing into 430 grains × 2048 samples each.

────────────────────────────────────────────────────────────
Health Tests: 0 low-var, 0 low-unique, 0 high-autocorr grains rejected.
────────────────────────────────────────────────────────────
METRIC               | RAW PCM    | LSB EXT    | MIXED/HASH
────────────────────────────────────────────────────────────
Shannon (bits/B)     | 6.5845 | 7.9983 | 7.9850
Min-Entropy (bits/B) | 2.8331 | 7.8057 | 7.4263
Bit Balance (1s)     | 0.4989 | 0.4991 | 0.4984
────────────────────────────────────────────────────────────
Per-Grain Shannon Entropy (Raw):
  Mean: 6.2215 bits/byte
  StdDev: 0.7409
────────────────────────────────────────────────────────────

[+] Saved entropy to 'entropy_output.bin'  (21.54s total)

E:\web\natrng\base_implementation>
```

autocorr threshold 0.95 -> 1