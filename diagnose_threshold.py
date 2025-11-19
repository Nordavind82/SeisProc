#!/usr/bin/env python3
"""Diagnose TF-Denoise thresholding logic."""
import numpy as np

# Simulate a spatial ensemble
print("="*60)
print("Diagnosing TF-Denoise Thresholding Logic")
print("="*60)

# Scenario 1: Ensemble with coherent signal
print("\n### Scenario 1: Coherent Signal Ensemble ###")
print("7 traces, all with signal at amplitude 100 + small noise")

spatial_amplitudes = np.array([
    98,   # Trace 1: signal + small noise
    102,  # Trace 2: signal + small noise
    99,   # Trace 3: signal + small noise
    101,  # Trace 4: signal + small noise (center trace)
    100,  # Trace 5: signal + small noise
    98,   # Trace 6: signal + small noise
    103   # Trace 7: signal + small noise
])

center_magnitude = 101  # Center trace amplitude

# Compute MAD threshold (as in the code)
median_amp = np.median(spatial_amplitudes)
mad = np.median(np.abs(spatial_amplitudes - median_amp))
threshold_k = 3.0
threshold = median_amp + threshold_k * mad

print(f"Spatial amplitudes: {spatial_amplitudes}")
print(f"Median: {median_amp:.2f}")
print(f"MAD: {mad:.2f}")
print(f"Threshold (median + {threshold_k}*MAD): {threshold:.2f}")
print(f"Center trace magnitude: {center_magnitude}")

# Apply soft thresholding
new_magnitude = max(center_magnitude - threshold, 0)
print(f"After soft threshold: max({center_magnitude} - {threshold:.2f}, 0) = {new_magnitude:.2f}")
print(f"Result: Signal amplitude {center_magnitude} → {new_magnitude:.2f} ({'REMOVED!' if new_magnitude < 10 else 'kept'})")

# Scenario 2: Ensemble with mostly noise, one signal
print("\n### Scenario 2: Noise Ensemble with One Signal ###")
print("7 traces: 6 with noise (amp ~5), 1 with signal (amp ~100)")

spatial_amplitudes = np.array([
    5,    # Trace 1: noise
    4,    # Trace 2: noise
    6,    # Trace 3: noise
    100,  # Trace 4: SIGNAL (center trace)
    5,    # Trace 5: noise
    5,    # Trace 6: noise
    4     # Trace 7: noise
])

center_magnitude = 100  # Center trace has signal

median_amp = np.median(spatial_amplitudes)
mad = np.median(np.abs(spatial_amplitudes - median_amp))
threshold = median_amp + threshold_k * mad

print(f"Spatial amplitudes: {spatial_amplitudes}")
print(f"Median: {median_amp:.2f}")
print(f"MAD: {mad:.2f}")
print(f"Threshold (median + {threshold_k}*MAD): {threshold:.2f}")
print(f"Center trace magnitude: {center_magnitude}")

new_magnitude = max(center_magnitude - threshold, 0)
print(f"After soft threshold: max({center_magnitude} - {threshold:.2f}, 0) = {new_magnitude:.2f}")
print(f"Result: Signal amplitude {center_magnitude} → {new_magnitude:.2f} ({'kept!' if new_magnitude > 50 else 'REDUCED'})")

# Scenario 3: Ensemble with noise only
print("\n### Scenario 3: Noise-Only Ensemble ###")
print("7 traces, all with noise (amp ~5)")

spatial_amplitudes = np.array([
    4,    # Trace 1: noise
    6,    # Trace 2: noise
    5,    # Trace 3: noise
    20,   # Trace 4: noise spike (center trace)
    5,    # Trace 5: noise
    4,    # Trace 6: noise
    6     # Trace 7: noise
])

center_magnitude = 20  # Center trace has noise spike

median_amp = np.median(spatial_amplitudes)
mad = np.median(np.abs(spatial_amplitudes - median_amp))
threshold = median_amp + threshold_k * mad

print(f"Spatial amplitudes: {spatial_amplitudes}")
print(f"Median: {median_amp:.2f}")
print(f"MAD: {mad:.2f}")
print(f"Threshold (median + {threshold_k}*MAD): {threshold:.2f}")
print(f"Center trace magnitude: {center_magnitude}")

new_magnitude = max(center_magnitude - threshold, 0)
print(f"After soft threshold: max({center_magnitude} - {threshold:.2f}, 0) = {new_magnitude:.2f}")
print(f"Result: Noise spike {center_magnitude} → {new_magnitude:.2f} ({'REMOVED!' if new_magnitude < 2 else 'kept'})")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)
print("Scenario 1 (coherent signal): Signal REMOVED - BAD!")
print("Scenario 2 (noise ensemble + signal): Signal kept - OK")
print("Scenario 3 (noise spike): Noise spike kept - BAD!")
print("\nPROBLEM: The threshold adapts to the ensemble content.")
print("With coherent signal, median is high → threshold high → signal removed!")
print("With random noise, median is low → threshold low → noise kept!")
print("\nThis is the OPPOSITE of what we want!")
print("="*60)
