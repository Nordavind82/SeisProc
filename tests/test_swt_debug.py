#!/usr/bin/env python3
"""
Debug test to trace SWT decomposition/reconstruction step by step.
"""

import numpy as np
import pywt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def trace_pywt_algorithm():
    """Trace through PyWavelets SWT algorithm step by step."""
    print("=" * 60)
    print("Tracing PyWavelets SWT Algorithm")
    print("=" * 60)

    wavelet = pywt.Wavelet('db4')
    dec_lo = np.array(wavelet.dec_lo)
    dec_hi = np.array(wavelet.dec_hi)
    rec_lo = np.array(wavelet.rec_lo)
    rec_hi = np.array(wavelet.rec_hi)
    flen = len(dec_lo)

    # Simple test signal
    n = 16  # Small for debugging
    signal = np.sin(np.linspace(0, 2*np.pi, n))

    print(f"\nInput signal ({n} samples):")
    print(signal)

    # Manual single-level decomposition
    center_shift = flen // 2
    level = 1
    step = 1

    print(f"\nFilter length: {flen}")
    print(f"Center shift: {center_shift}")

    # Decomposition
    cA = np.zeros(n)
    cD = np.zeros(n)

    print("\nDecomposition (formula: idx = (i - k*step + center_shift) % n):")
    for i in range(min(3, n)):  # Just show first 3 samples
        print(f"\n  Output[{i}]:")
        lo_sum = 0.0
        hi_sum = 0.0
        for k in range(flen):
            idx = (i - k * step + center_shift) % n
            val = signal[idx]
            lo_sum += val * dec_lo[k]
            hi_sum += val * dec_hi[k]
            if k < 4:  # Show first 4 filter taps
                print(f"    k={k}: idx={idx}, signal[{idx}]={val:.4f}, "
                      f"dec_lo[{k}]={dec_lo[k]:.4f}, dec_hi[{k}]={dec_hi[k]:.4f}")
        cA[i] = lo_sum
        cD[i] = hi_sum
        print(f"    -> cA[{i}]={lo_sum:.4f}, cD[{i}]={hi_sum:.4f}")

    # Complete decomposition
    for i in range(n):
        for k in range(flen):
            idx = (i - k * step + center_shift) % n
            cA[i] += signal[idx] * dec_lo[k]
            cD[i] += signal[idx] * dec_hi[k]
        # Subtract what we added above for i < 3
        if i < 3:
            cA[i] /= 2
            cD[i] /= 2

    # Compare with PyWavelets
    coeffs = pywt.swt(signal, 'db4', level=1)
    cA_pywt, cD_pywt = coeffs[0]

    print("\n\nCoefficient Comparison:")
    print("=" * 60)
    print(f"{'i':>3} | {'Manual cA':>12} | {'PyWt cA':>12} | {'Diff':>10}")
    print("-" * 50)
    for i in range(n):
        diff = cA[i] - cA_pywt[i]
        print(f"{i:3d} | {cA[i]:12.6f} | {cA_pywt[i]:12.6f} | {diff:10.6f}")

    print("\nManual cD vs PyWavelets cD:")
    print(f"{'i':>3} | {'Manual cD':>12} | {'PyWt cD':>12} | {'Diff':>10}")
    print("-" * 50)
    for i in range(n):
        diff = cD[i] - cD_pywt[i]
        print(f"{i:3d} | {cD[i]:12.6f} | {cD_pywt[i]:12.6f} | {diff:10.6f}")

    # Reconstruction
    print("\n\nReconstruction (formula: idx = (i + k*step - center_shift) % n, reversed filters):")
    rec_lo_rev = rec_lo[::-1]
    rec_hi_rev = rec_hi[::-1]

    output = np.zeros(n)
    for i in range(min(3, n)):
        print(f"\n  Output[{i}]:")
        recon_sum = 0.0
        for k in range(flen):
            idx = (i + k * step - center_shift) % n
            a_val = cA_pywt[idx]  # Use PyWt coeffs for comparison
            d_val = cD_pywt[idx]
            recon_sum += a_val * rec_lo_rev[k] + d_val * rec_hi_rev[k]
            if k < 4:
                print(f"    k={k}: idx={idx}, cA={a_val:.4f}, cD={d_val:.4f}, "
                      f"rec_lo_rev[{k}]={rec_lo_rev[k]:.4f}")
        output[i] = recon_sum * 0.5
        print(f"    -> output[{i}]={output[i]:.4f} (after *0.5)")

    # Complete reconstruction using PyWavelets coefficients
    for i in range(n):
        recon_sum = 0.0
        for k in range(flen):
            idx = (i + k * step - center_shift) % n
            recon_sum += cA_pywt[idx] * rec_lo_rev[k] + cD_pywt[idx] * rec_hi_rev[k]
        output[i] = recon_sum * 0.5

    # Compare with input and PyWavelets reconstruction
    recon_pywt = pywt.iswt(coeffs, 'db4')

    print("\n\nReconstruction Comparison:")
    print("=" * 60)
    print(f"{'i':>3} | {'Input':>10} | {'Manual':>10} | {'PyWt':>10} | {'Man-In':>10}")
    print("-" * 60)
    for i in range(n):
        diff = output[i] - signal[i]
        print(f"{i:3d} | {signal[i]:10.4f} | {output[i]:10.4f} | {recon_pywt[i]:10.4f} | {diff:10.6f}")

    print(f"\nManual reconstruction max error: {np.max(np.abs(output - signal)):.2e}")
    print(f"PyWavelets reconstruction max error: {np.max(np.abs(recon_pywt - signal)):.2e}")


def check_cpp_center_shift():
    """Check if C++ center shift matches PyWavelets."""
    print("\n\n" + "=" * 60)
    print("Checking C++ Algorithm Against PyWavelets")
    print("=" * 60)

    # PyWavelets SWT uses a different center alignment
    # Let's check what happens with different center_shift values

    wavelet = pywt.Wavelet('db4')
    dec_lo = np.array(wavelet.dec_lo)
    dec_hi = np.array(wavelet.dec_hi)
    flen = len(dec_lo)

    n = 16
    signal = np.sin(np.linspace(0, 2*np.pi, n))

    # Get PyWavelets result
    coeffs = pywt.swt(signal, 'db4', level=1)
    cA_pywt, cD_pywt = coeffs[0]

    print("\nTrying different center_shift values to match PyWavelets:")
    print("-" * 60)

    for center_shift in range(flen + 2):
        cA_test = np.zeros(n)
        for i in range(n):
            for k in range(flen):
                idx = (i - k + center_shift) % n
                cA_test[i] += signal[idx] * dec_lo[k]

        diff = np.max(np.abs(cA_test - cA_pywt))
        if diff < 0.01:
            print(f"  center_shift={center_shift}: max_diff={diff:.6f} *** MATCH ***")
        else:
            print(f"  center_shift={center_shift}: max_diff={diff:.6f}")

    # Also try without any center shift
    print("\n\nTrying formula WITHOUT center shift:")
    for offset in range(-2, 10):
        cA_test = np.zeros(n)
        for i in range(n):
            for k in range(flen):
                idx = (i - k + offset) % n
                cA_test[i] += signal[idx] * dec_lo[k]

        diff = np.max(np.abs(cA_test - cA_pywt))
        if diff < 0.01:
            print(f"  offset={offset}: max_diff={diff:.6f} *** MATCH ***")

    # PyWavelets uses mode='periodization' internally which may have different alignment
    # Let's also check what happens if we use different index formulas
    print("\n\nTrying different index formulas:")

    # Formula 1: idx = (i - k + flen//2) % n (our current formula)
    cA1 = np.zeros(n)
    for i in range(n):
        for k in range(flen):
            idx = (i - k + flen//2) % n
            cA1[i] += signal[idx] * dec_lo[k]

    # Formula 2: idx = (i - k * step) % n (no center shift)
    cA2 = np.zeros(n)
    for i in range(n):
        for k in range(flen):
            idx = (i - k) % n
            cA2[i] += signal[idx] * dec_lo[k]

    # Formula 3: idx = (i + k) % n
    cA3 = np.zeros(n)
    for i in range(n):
        for k in range(flen):
            idx = (i + k) % n
            cA3[i] += signal[idx] * dec_lo[k]

    # Formula 4: reversed filter order
    cA4 = np.zeros(n)
    for i in range(n):
        for k in range(flen):
            idx = (i - k + flen//2) % n
            cA4[i] += signal[idx] * dec_lo[flen - 1 - k]  # reversed

    print(f"Formula 1 (i - k + flen//2) diff: {np.max(np.abs(cA1 - cA_pywt)):.6f}")
    print(f"Formula 2 (i - k) diff: {np.max(np.abs(cA2 - cA_pywt)):.6f}")
    print(f"Formula 3 (i + k) diff: {np.max(np.abs(cA3 - cA_pywt)):.6f}")
    print(f"Formula 4 (reversed filter) diff: {np.max(np.abs(cA4 - cA_pywt)):.6f}")


if __name__ == "__main__":
    trace_pywt_algorithm()
    check_cpp_center_shift()
