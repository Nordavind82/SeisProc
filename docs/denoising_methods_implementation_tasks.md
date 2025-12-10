# Denoising Methods Implementation Tasks

## Overview

Implementation tasks for adding advanced denoising methods to SeisProc's time-frequency processors. These methods complement and enhance the existing MAD-based thresholding.

---

## Phase 1: Quick Wins (Add to Existing Processors)

### Task D1: SURE Threshold Selection

**Priority**: High
**Effort**: 2 hours
**Adds to**: All TF processors (TFDenoise, DWTDenoise, future Gabor, etc.)

#### Description
Implement Stein's Unbiased Risk Estimate (SURE) for automatic threshold selection, eliminating the need for manual k parameter tuning.

#### Subtasks

- [ ] **D1.1** Create `processors/thresholding.py` module
  - Implement `sure_threshold(coeffs, sigma)` function
  - Implement SURE criterion computation
  - Add optimization to find optimal threshold

- [ ] **D1.2** Add SURE option to DWTDenoise
  - Add `threshold_method` parameter: 'mad', 'sure', 'bayes'
  - Update `_process_dwt()` to use selected method
  - Update `_process_swt()` similarly

- [ ] **D1.3** Add SURE option to TFDenoise
  - Add `threshold_method` parameter
  - Update STFT and S-Transform processing

- [ ] **D1.4** Update UI
  - Add "Threshold Method" dropdown to DWT and TFD parameter groups
  - Options: "MAD (Manual k)", "SURE (Automatic)", "BayesShrink (Automatic)"

- [ ] **D1.5** Create tests
  - Test SURE threshold computation
  - Compare auto vs manual threshold quality
  - Verify parameter-free operation

#### Implementation Notes
```python
def sure_threshold(coeffs, sigma):
    """
    Compute SURE-optimal soft threshold.

    SURE(λ) = -n + 2×#{i: |y_i| ≤ λ} + Σ min(y_i², λ²)
    """
    n = len(coeffs)
    sorted_coeffs = np.sort(np.abs(coeffs))

    # Compute SURE for each potential threshold
    sure_values = []
    for i, lam in enumerate(sorted_coeffs):
        num_below = i + 1
        risk = -n + 2 * num_below
        risk += np.sum(np.minimum(coeffs**2, lam**2))
        sure_values.append(risk)

    # Return threshold minimizing SURE
    best_idx = np.argmin(sure_values)
    return sorted_coeffs[best_idx]
```

---

### Task D2: BayesShrink Threshold

**Priority**: High
**Effort**: 2 hours
**Adds to**: DWTDenoise, WPT (future)

#### Description
Implement BayesShrink for level-dependent automatic thresholding in wavelet transforms.

#### Subtasks

- [ ] **D2.1** Add to `processors/thresholding.py`
  - Implement `bayesshrink_threshold(coeffs, sigma_noise)` function
  - Handle edge case where signal variance ≈ 0

- [ ] **D2.2** Integrate with DWTDenoise
  - Apply different threshold per decomposition level
  - Estimate noise sigma from finest level

- [ ] **D2.3** Update processing methods
  - Modify `_process_dwt()` for level-dependent thresholding
  - Modify `_process_swt()` similarly

- [ ] **D2.4** Create tests
  - Test level-dependent behavior
  - Compare with universal threshold

#### Implementation Notes
```python
def bayesshrink_threshold(coeffs, sigma_noise):
    """
    BayesShrink threshold: λ = σ²_noise / σ_signal
    """
    sigma_coef = np.std(coeffs)
    sigma_signal_sq = max(sigma_coef**2 - sigma_noise**2, 0)

    if sigma_signal_sq == 0:
        # Pure noise - remove all
        return np.max(np.abs(coeffs))

    sigma_signal = np.sqrt(sigma_signal_sq)
    return sigma_noise**2 / sigma_signal
```

---

### Task D3: Block Thresholding

**Priority**: High
**Effort**: 4 hours
**Adds to**: TFDenoise (STFT), future Gabor

#### Description
Implement block thresholding (BlockJS) to threshold groups of coefficients together, preserving local structure.

#### Subtasks

- [ ] **D3.1** Add to `processors/thresholding.py`
  - Implement `block_threshold(coeffs, block_size, threshold)` function
  - Implement block energy computation
  - Handle edge blocks (padding or truncation)

- [ ] **D3.2** Add 2D block thresholding for TF matrices
  - Implement `block_threshold_2d(tf_matrix, block_shape, threshold)`
  - Support rectangular blocks (time × frequency)

- [ ] **D3.3** Integrate with TFDenoise STFT mode
  - Add `use_block_threshold` parameter
  - Add `block_size` parameter

- [ ] **D3.4** Update UI
  - Add "Block Threshold" checkbox to TFD parameters
  - Add block size controls (time bins, freq bins)

- [ ] **D3.5** Create tests
  - Test block shrinkage formula
  - Compare SNR with point-wise thresholding
  - Test edge handling

#### Implementation Notes
```python
def block_threshold_2d(tf_matrix, block_shape, threshold):
    """
    Block James-Stein thresholding for TF matrices.

    Shrink factor: (1 - λ²/||block||²)₊
    """
    result = np.zeros_like(tf_matrix)
    bt, bf = block_shape

    for i in range(0, tf_matrix.shape[0], bt):
        for j in range(0, tf_matrix.shape[1], bf):
            block = tf_matrix[i:i+bt, j:j+bf]
            block_energy = np.sum(np.abs(block)**2)

            if block_energy > 0:
                shrink = max(1 - threshold**2 / block_energy, 0)
                result[i:i+bt, j:j+bf] = shrink * block

    return result
```

---

## Phase 2: Enhanced Methods

### Task D4: Bivariate Shrinkage

**Priority**: Medium
**Effort**: 4 hours
**Adds to**: DWTDenoise

#### Description
Implement bivariate shrinkage using parent-child coefficient relationships in wavelet tree for superior edge preservation.

#### Subtasks

- [ ] **D4.1** Add to `processors/thresholding.py`
  - Implement `bivariate_shrink(child, parent, sigma)` function
  - Handle coefficient alignment between levels

- [ ] **D4.2** Create parent-child mapping function
  - Map each coefficient to its parent in coarser level
  - Handle boundary conditions

- [ ] **D4.3** Integrate with DWTDenoise
  - Add `use_bivariate` parameter
  - Modify `_process_dwt()` to use parent coefficients

- [ ] **D4.4** Update UI
  - Add "Bivariate Shrinkage" checkbox to DWT parameters
  - Add tooltip explaining the method

- [ ] **D4.5** Create tests
  - Test edge preservation (step function, impulse)
  - Compare with standard soft thresholding
  - Verify parent-child mapping

#### Implementation Notes
```python
def bivariate_shrink(child_coeffs, parent_coeffs, sigma):
    """
    Bivariate shrinkage using parent coefficient.

    w_out = √(max(s² - 3σ², 0)) / s × w
    where s² = w² + w_parent²
    """
    # Upsample parent to match child size
    parent_upsampled = np.repeat(parent_coeffs, 2)[:len(child_coeffs)]

    s_squared = child_coeffs**2 + parent_upsampled**2
    s = np.sqrt(s_squared)

    sigma_signal = np.sqrt(np.maximum(s_squared - 3*sigma**2, 0))

    # Avoid division by zero
    shrink_factor = np.where(s > 0, sigma_signal / s, 0)

    return shrink_factor * child_coeffs
```

---

### Task D5: Adaptive Local MAD

**Priority**: Medium
**Effort**: 1 day
**Adds to**: All TF processors

#### Description
Implement spatially-varying threshold based on local noise statistics for non-stationary noise handling.

#### Subtasks

- [ ] **D5.1** Add to `processors/thresholding.py`
  - Implement `compute_local_mad(matrix, window_size)` function
  - Use efficient median filter from scipy.ndimage

- [ ] **D5.2** Create adaptive threshold map
  - Implement `adaptive_threshold_map(tf_matrix, k, window_size)`
  - Handle edges appropriately

- [ ] **D5.3** Integrate with TFDenoise
  - Add `adaptive_threshold` parameter
  - Add `adaptation_window` parameter (time, frequency)

- [ ] **D5.4** Integrate with DWTDenoise
  - Apply adaptive threshold per coefficient group
  - Use spatial aperture for adaptation

- [ ] **D5.5** Update UI
  - Add "Adaptive Threshold" checkbox
  - Add window size controls

- [ ] **D5.6** Create tests
  - Test with non-stationary noise
  - Verify local adaptation behavior
  - Compare with global threshold

#### Implementation Notes
```python
def adaptive_threshold_map(tf_matrix, k, window_size):
    """
    Compute spatially-varying threshold based on local MAD.
    """
    from scipy.ndimage import median_filter

    abs_matrix = np.abs(tf_matrix)

    # Local median
    local_median = median_filter(abs_matrix, size=window_size)

    # Local MAD
    local_mad = median_filter(
        np.abs(abs_matrix - local_median),
        size=window_size
    )

    # Scale for Gaussian consistency
    return k * 1.4826 * local_mad
```

---

### Task D6: Additional Threshold Functions

**Priority**: Low
**Effort**: 2 hours
**Adds to**: All processors

#### Description
Add firm and garrote thresholding functions as alternatives to soft/hard.

#### Subtasks

- [ ] **D6.1** Add to `processors/thresholding.py`
  - Implement `garrote_threshold(x, lam)`
  - Implement `firm_threshold(x, lam1, lam2)`

- [ ] **D6.2** Update threshold mode options
  - Add 'garrote', 'firm' to threshold_mode parameter

- [ ] **D6.3** Update UI
  - Add new options to threshold mode dropdowns

- [ ] **D6.4** Create tests
  - Test continuity of firm threshold
  - Compare bias of different functions

---

## Phase 3: Advanced Processors

### Task D7: RPCA Denoising Processor

**Priority**: Medium
**Effort**: 1-2 days
**New processor**: `processors/rpca_denoise.py`

#### Description
Implement Robust PCA for decomposing time-frequency representation into low-rank (signal) and sparse (noise) components.

#### Subtasks

- [ ] **D7.1** Create `processors/rpca_denoise.py`
  - Implement `RPCADenoise` class
  - Implement ADMM solver for L + S decomposition
  - Parameters: lambda (sparsity weight), max_iter, tolerance

- [ ] **D7.2** Implement core RPCA algorithm
  - Singular value thresholding for L
  - Soft thresholding for S
  - Convergence checking

- [ ] **D7.3** Add processing modes
  - 2D: Process entire gather as matrix
  - Sliding window: Process overlapping windows

- [ ] **D7.4** UI Integration
  - Add "RPCA Decomposition" to algorithm dropdown
  - Create parameter group with lambda, iterations

- [ ] **D7.5** Create tests
  - Test low-rank recovery
  - Test sparse noise separation
  - Benchmark convergence speed

#### Implementation Notes
```python
class RPCADenoise(BaseProcessor):
    def __init__(self, lambda_param=None, max_iter=100, tol=1e-6):
        self.lambda_param = lambda_param  # Auto if None
        self.max_iter = max_iter
        self.tol = tol

    def _rpca_admm(self, M):
        """ADMM solver for min ||L||_* + λ||S||_1 s.t. M = L + S"""
        if self.lambda_param is None:
            self.lambda_param = 1 / np.sqrt(max(M.shape))

        L = np.zeros_like(M)
        S = np.zeros_like(M)
        Y = np.zeros_like(M)  # Dual variable
        mu = 1.0

        for _ in range(self.max_iter):
            # Update L (singular value thresholding)
            U, s, Vt = np.linalg.svd(M - S + Y/mu, full_matrices=False)
            s_thresh = np.maximum(s - 1/mu, 0)
            L = U @ np.diag(s_thresh) @ Vt

            # Update S (soft thresholding)
            S = soft_threshold(M - L + Y/mu, self.lambda_param/mu)

            # Update dual
            Y = Y + mu * (M - L - S)

            # Check convergence
            if np.linalg.norm(M - L - S) < self.tol:
                break

        return L  # Return low-rank (denoised) component
```

---

### Task D8: Non-Local Means in TF Domain

**Priority**: Low
**Effort**: 2 days
**New processor**: `processors/nlm_denoise.py`

#### Description
Implement Non-Local Means denoising operating in time-frequency domain for preserving repetitive structures.

#### Subtasks

- [ ] **D8.1** Create `processors/nlm_denoise.py`
  - Implement `NLMDenoise` class
  - Parameters: patch_size, search_window, h (smoothing)

- [ ] **D8.2** Implement patch similarity computation
  - Efficient patch extraction
  - Gaussian-weighted L2 distance
  - Fast search with approximate NN (optional)

- [ ] **D8.3** Implement weighted averaging
  - Compute weights from similarities
  - Apply weighted average to center pixel

- [ ] **D8.4** Optimize for performance
  - Use vectorized operations
  - Consider GPU acceleration (PyTorch)

- [ ] **D8.5** UI Integration
  - Add to algorithm dropdown
  - Parameter controls for patch size, h

- [ ] **D8.6** Create tests
  - Test repetitive structure preservation
  - Benchmark performance
  - Compare with local methods

---

### Task D9: TV Regularization Denoising

**Priority**: Low
**Effort**: 1 day
**New processor**: `processors/tv_denoise.py`

#### Description
Implement Total Variation regularization for edge-preserving denoising.

#### Subtasks

- [ ] **D9.1** Create `processors/tv_denoise.py`
  - Implement `TVDenoise` class
  - Use split-Bregman or Chambolle algorithm
  - Parameters: lambda (regularization weight), max_iter

- [ ] **D9.2** Implement TV minimization
  - Isotropic or anisotropic TV
  - Efficient iterative solver

- [ ] **D9.3** Apply to seismic data
  - Process gather as 2D image
  - Or apply to TF representation

- [ ] **D9.4** UI and tests
  - Add to dropdown
  - Test edge preservation

---

## Summary Table

| Task | Method | Priority | Effort | Adds To |
|------|--------|----------|--------|---------|
| D1 | SURE Threshold | High | 2 hrs | All TF processors |
| D2 | BayesShrink | High | 2 hrs | DWT |
| D3 | Block Threshold | High | 4 hrs | STFT/Gabor |
| D4 | Bivariate Shrink | Medium | 4 hrs | DWT |
| D5 | Adaptive MAD | Medium | 1 day | All |
| D6 | Garrote/Firm | Low | 2 hrs | All |
| D7 | RPCA | Medium | 1-2 days | New processor |
| D8 | NLM | Low | 2 days | New processor |
| D9 | TV Regularization | Low | 1 day | New processor |

---

## Implementation Order

### Recommended Sequence

1. **D1 + D2**: SURE and BayesShrink (automatic thresholding)
2. **D3**: Block thresholding (improved STFT quality)
3. **D4**: Bivariate shrinkage (improved DWT quality)
4. **D5**: Adaptive MAD (non-stationary noise)
5. **D7**: RPCA (new capability for coherent noise)
6. **D6, D8, D9**: Additional methods as needed

### Dependencies

```
D1 (SURE) ─────────────────────────┐
D2 (BayesShrink) ──────────────────┼──> Enhanced automatic thresholding
D3 (Block) ────────────────────────┘

D4 (Bivariate) ──────> Requires D1/D2 for sigma estimation

D5 (Adaptive MAD) ──────> Can use D1/D2/D3 locally

D7 (RPCA) ──────> Independent, new processor
D8 (NLM) ───────> Independent, new processor
D9 (TV) ────────> Independent, new processor
```

---

## Testing Checklist

For each new method:

- [ ] Unit test for algorithm correctness
- [ ] SNR improvement test vs baseline MAD
- [ ] Edge/transient preservation test
- [ ] Performance benchmark
- [ ] Integration test with UI
- [ ] Regression test (existing functionality unchanged)
