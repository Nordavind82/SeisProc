# Denoising Methods for Time-Frequency Decomposition

## Overview

This document outlines state-of-the-art denoising techniques that can be applied to time-frequency decompositions in SeisProc. These methods go beyond the current MAD (Median Absolute Deviation) thresholding to provide better signal preservation and noise removal.

## Current Implementation

**Method**: MAD-based thresholding with soft/hard shrinkage
**Formula**: `threshold = k × 1.4826 × median(|x - median(x)|)`
**Pros**: Robust to outliers, simple, fast
**Cons**: Single parameter (k), not adaptive to local conditions

---

## Thresholding Methods

### 1. Statistical Threshold Estimation Methods

#### 1.1 Universal Threshold (VisuShrink)
```
λ = σ × √(2 × log(n))
```
- **Author**: Donoho & Johnstone (1994)
- **Description**: Theoretically optimal for Gaussian white noise
- **Pros**:
  - Simple closed-form formula
  - Guaranteed to remove noise with high probability
  - Works well for smooth signals
- **Cons**:
  - Tends to over-smooth
  - Assumes constant noise level
  - Can remove fine details
- **Best for**: Initial denoising pass, very noisy data
- **Implementation**: `threshold = sigma * np.sqrt(2 * np.log(n))`

#### 1.2 SURE Threshold (SureShrink)
```
λ = argmin SURE(λ) where SURE estimates MSE without knowing true signal
```
- **Author**: Donoho & Johnstone (1995)
- **Description**: Minimizes Stein's Unbiased Risk Estimate
- **Pros**:
  - Adaptive - finds optimal threshold automatically
  - No manual parameter tuning
  - Theoretically optimal for Gaussian noise
- **Cons**:
  - Assumes Gaussian noise distribution
  - Slightly more compute than universal
- **Best for**: Automatic threshold selection, production pipelines
- **Implementation**:
  ```python
  # PyWavelets has built-in SURE
  threshold = pywt.threshold_firm(coeffs, value_low, value_high)
  # Or compute SURE criterion and minimize
  ```

#### 1.3 BayesShrink
```
λ = σ² / σ_signal  where σ_signal = √(max(σ²_coef - σ², 0))
```
- **Author**: Chang, Yu & Vetterli (2000)
- **Description**: Bayesian estimation assuming Generalized Gaussian prior
- **Pros**:
  - Level-dependent thresholding (different λ per wavelet level)
  - Better detail preservation than universal
  - Adapts to signal characteristics at each scale
- **Cons**:
  - Requires variance estimation at each level
  - Assumes GGD signal prior
- **Best for**: Wavelet denoising (DWT, SWT, WPT)
- **Implementation**:
  ```python
  def bayesshrink_threshold(coeffs, sigma_noise):
      sigma_coef = np.std(coeffs)
      sigma_signal = np.sqrt(max(sigma_coef**2 - sigma_noise**2, 0))
      if sigma_signal == 0:
          return np.max(np.abs(coeffs))  # Remove all
      return sigma_noise**2 / sigma_signal
  ```

#### 1.4 Minimax Threshold
```
λ = σ × λ_n  where λ_n is tabulated minimax value
```
- **Description**: Minimizes maximum risk over signal class
- **Pros**: Robust worst-case performance
- **Cons**: Conservative, may under-denoise
- **Best for**: Safety-critical applications

#### 1.5 Cross-Validation Threshold
```
λ = argmin CV(λ) using leave-one-out or k-fold
```
- **Description**: Data-driven threshold selection
- **Pros**: Optimal for specific dataset
- **Cons**: Computationally expensive, may overfit
- **Best for**: Offline processing, parameter tuning

---

### 2. Thresholding Functions

#### 2.1 Hard Thresholding
```
η_hard(x, λ) = x if |x| > λ else 0
```
- **Behavior**: Keep coefficient unchanged or set to zero
- **Pros**: Preserves large coefficients exactly
- **Cons**: Creates discontinuity at threshold, can cause artifacts
- **Best for**: Sparse signals, preserving sharp edges

#### 2.2 Soft Thresholding (Current Default)
```
η_soft(x, λ) = sign(x) × max(|x| - λ, 0)
```
- **Behavior**: Shrink all coefficients toward zero
- **Pros**: Continuous, no artifacts, smooth result
- **Cons**: Biases large coefficients, can over-smooth
- **Best for**: General denoising, smooth signals

#### 2.3 Garrote (Non-negative Garrote)
```
η_garrote(x, λ) = x - λ²/x if |x| > λ else 0
```
- **Behavior**: Compromise between hard and soft
- **Pros**: Less bias than soft, smoother than hard
- **Cons**: More complex, undefined at x=0
- **Best for**: Balance between edge preservation and smoothness

#### 2.4 SCAD (Smoothly Clipped Absolute Deviation)
```
η_SCAD(x, λ, a) = piecewise function with smooth clipping
```
- **Behavior**: Nearly unbiased for large coefficients
- **Pros**: Oracle properties, minimal bias
- **Cons**: Two parameters (λ, a), complex
- **Best for**: Sparse regression, high SNR

#### 2.5 Firm Thresholding
```
η_firm(x, λ1, λ2) = 0 if |x| ≤ λ1
                  = sign(x) × λ2(|x| - λ1)/(λ2 - λ1) if λ1 < |x| ≤ λ2
                  = x if |x| > λ2
```
- **Behavior**: Linear transition zone between hard regions
- **Pros**: Tuneable compromise, continuous
- **Cons**: Two parameters
- **Best for**: Fine control over threshold behavior

---

## Advanced Denoising Techniques

### 3. Block Thresholding

#### 3.1 BlockJS (Block James-Stein)
```
Shrink blocks of coefficients: y_B = (1 - λ/||y_B||²)₊ × y_B
```
- **Description**: Threshold groups of coefficients together rather than individually
- **Rationale**: Adjacent coefficients are often correlated (wavelet locality)
- **Block size**: Typically √n or log(n)
- **Pros**:
  - Preserves local structure
  - Better for correlated signals
  - Improved visual quality
- **Cons**:
  - Block size selection
  - Edge effects at block boundaries
- **Seismic benefit**: Preserves wavelet shape, better reflector continuity
- **Implementation**:
  ```python
  def block_threshold(coeffs, block_size, threshold):
      n_blocks = len(coeffs) // block_size
      result = np.zeros_like(coeffs)
      for i in range(n_blocks):
          block = coeffs[i*block_size:(i+1)*block_size]
          block_energy = np.sum(block**2)
          shrink_factor = max(1 - threshold/block_energy, 0)
          result[i*block_size:(i+1)*block_size] = shrink_factor * block
      return result
  ```

#### 3.2 NeighBlock
- **Description**: Overlapping blocks with weighted averaging
- **Pros**: Reduces block boundary artifacts
- **Cons**: More computation

### 4. Bivariate Shrinkage

```
Estimate coefficient using parent: w = √(max(s² - 3σ², 0)) / s × w
where s² = w² + w_parent²
```
- **Author**: Sendur & Selesnick (2002)
- **Description**: Exploit parent-child relationships in wavelet tree
- **Rationale**: Large parent coefficient suggests signal, not noise
- **Pros**:
  - Exploits inter-scale dependencies
  - Better edge preservation
  - Reduces false positives in smooth regions
- **Cons**:
  - Only for hierarchical transforms (DWT, WPT)
  - More complex implementation
- **Seismic benefit**: Better first break preservation, sharp event edges
- **Implementation**:
  ```python
  def bivariate_shrink(child, parent, sigma):
      s = np.sqrt(child**2 + parent**2)
      sigma_signal = np.sqrt(max(s**2 - 3*sigma**2, 0))
      if s == 0:
          return 0
      return sigma_signal / s * child
  ```

### 5. Context Modeling / Adaptive Thresholding

```
λ(x, y) = k × MAD_local(x, y)  # Spatially varying threshold
```
- **Description**: Compute local statistics in sliding windows
- **Rationale**: Noise level often varies across data
- **Window size**: Typically 16-64 samples, 5-15 traces
- **Pros**:
  - Adapts to non-stationary noise
  - Better for field data with varying conditions
- **Cons**:
  - Window size selection
  - Edge handling
- **Seismic benefit**: Handles AVO effects, varying noise with offset
- **Implementation**:
  ```python
  def adaptive_threshold(tf_matrix, window_size, k):
      from scipy.ndimage import uniform_filter
      local_median = median_filter(np.abs(tf_matrix), size=window_size)
      local_mad = median_filter(np.abs(tf_matrix - local_median), size=window_size)
      return k * 1.4826 * local_mad
  ```

---

## State-of-the-Art Methods

### 6. Non-Local Means (NLM) in Transform Domain

```
y_i = Σ_j w(i,j) × x_j  where w(i,j) ∝ exp(-||P_i - P_j||² / h²)
```
- **Description**: Average similar patches, not just spatial neighbors
- **Key idea**: Find similar patterns anywhere in the data
- **Patch size**: Typically 5×5 to 11×11 in TF domain
- **Search window**: Local or global
- **Pros**:
  - Preserves repetitive structures
  - Excellent for periodic/quasi-periodic signals
  - State-of-the-art for image denoising
- **Cons**:
  - Computationally expensive O(n² × patch_size)
  - Parameter tuning (h, patch size, search window)
- **Seismic benefit**: Excellent for multiples, coherent reflections
- **Implementation**: Use `skimage.restoration.denoise_nl_means` as reference

### 7. Low-Rank Matrix Approximation

#### 7.1 Truncated SVD
```
A ≈ U_r × S_r × V_r^T  (keep top r singular values)
```
- **Description**: Approximate TF matrix with low-rank matrix
- **Pros**: Simple, fast with randomized SVD
- **Cons**: Hard threshold on rank

#### 7.2 Robust PCA (RPCA)
```
min ||L||_* + λ||S||_1  subject to  A = L + S
```
- **Description**: Decompose into Low-rank (signal) + Sparse (noise/outliers)
- **Solver**: ADMM, ALM, or proximal gradient
- **Pros**:
  - Separates structured signal from random noise
  - Handles outliers/spikes
  - Theoretically principled
- **Cons**:
  - Iterative solver, slower
  - λ parameter selection
- **Seismic benefit**: Multiples often low-rank, random noise is sparse
- **Implementation**:
  ```python
  def rpca_denoise(tf_matrix, lambda_param=None):
      if lambda_param is None:
          lambda_param = 1 / np.sqrt(max(tf_matrix.shape))
      # ADMM solver for L + S decomposition
      # Returns low-rank component L as denoised signal
  ```

#### 7.3 Nuclear Norm Minimization
```
min ||X||_* subject to ||A - X||_F ≤ ε
```
- **Description**: Find matrix with minimum nuclear norm close to data
- **Pros**: Convex relaxation of rank minimization
- **Cons**: Requires noise level estimate ε

### 8. Total Variation (TV) Regularization

```
min ||y - x||² + λ × TV(x)  where TV(x) = Σ|∇x|
```
- **Description**: Minimize signal variation while fitting data
- **Pros**:
  - Preserves sharp edges
  - Removes oscillatory noise
  - Well-understood theory
- **Cons**:
  - Staircasing artifacts in smooth regions
  - λ selection
- **Seismic benefit**: Sharp reflector boundaries
- **Variants**: TV-L1, TV-L2, Anisotropic TV
- **Implementation**: `skimage.restoration.denoise_tv_chambolle`

### 9. Sparse Coding / Dictionary Learning

```
min ||x - Dα||² + λ||α||_1
```
- **Description**: Learn optimal dictionary D from data, represent signal sparsely
- **Training**: K-SVD, Online Dictionary Learning
- **Pros**:
  - Data-adaptive basis functions
  - Can learn seismic wavelet shapes
  - State-of-the-art for natural images
- **Cons**:
  - Requires training phase
  - Computationally expensive
  - May not generalize across datasets
- **Seismic potential**: Learn dictionaries for specific wavelet types

---

## Deep Learning Approaches

### 10. Denoising Autoencoders
- **Architecture**: Encoder → bottleneck → Decoder
- **Training**: Input noisy, target clean
- **Pros**: Can learn complex noise patterns
- **Cons**: Needs paired training data

### 11. DnCNN (Denoising CNN)
- **Architecture**: Residual learning - predict noise, subtract
- **Key**: Batch normalization, residual learning
- **Pros**: State-of-the-art for known noise levels
- **Cons**: Needs retraining for different noise levels

### 12. U-Net for Seismic
- **Architecture**: Encoder-decoder with skip connections
- **Pros**: Multi-scale processing, good for structured data
- **Cons**: Large model, training data requirements

### 13. Self-Supervised Methods

#### Noise2Noise
- **Key insight**: Can train with only noisy pairs (no clean data)
- **Requirement**: Two independent noisy observations of same scene

#### Noise2Void
- **Key insight**: Predict pixel from neighbors only
- **Pros**: Single noisy image sufficient
- **Seismic potential**: Very promising for field data

---

## Comparison Matrix

| Method | Compute | Quality | Auto-tune | Edge Preserve | Seismic Fit |
|--------|---------|---------|-----------|---------------|-------------|
| MAD + Soft (current) | ★★★★★ | ★★★ | No | ★★ | ★★★ |
| SURE | ★★★★★ | ★★★★ | **Yes** | ★★★ | ★★★★ |
| BayesShrink | ★★★★★ | ★★★★ | **Yes** | ★★★ | ★★★★ |
| Block Threshold | ★★★★ | ★★★★ | Partial | ★★★★ | ★★★★ |
| Bivariate Shrink | ★★★★ | ★★★★★ | Partial | ★★★★★ | ★★★★★ |
| Adaptive MAD | ★★★★ | ★★★★ | Partial | ★★★ | ★★★★★ |
| NLM | ★★ | ★★★★★ | No | ★★★★★ | ★★★★★ |
| RPCA | ★★★ | ★★★★★ | Partial | ★★★★ | ★★★★ |
| TV Regularization | ★★★ | ★★★★ | No | ★★★★★ | ★★★★ |
| Deep Learning | ★ | ★★★★★ | N/A | ★★★★★ | ★★★ |

---

## Implementation Priority

### Phase 1: Quick Wins (Add to Existing Processors)

| Priority | Method | Effort | Impact |
|----------|--------|--------|--------|
| 1 | SURE threshold | 2 hrs | Auto threshold, no k tuning |
| 2 | BayesShrink | 2 hrs | Level-adaptive for DWT |
| 3 | Block threshold | 4 hrs | Better wavelet preservation |

### Phase 2: Enhanced Methods

| Priority | Method | Effort | Impact |
|----------|--------|--------|--------|
| 4 | Bivariate shrinkage | 4 hrs | Excellent edge preservation |
| 5 | Adaptive local MAD | 1 day | Non-stationary noise |
| 6 | Firm/Garrote threshold | 2 hrs | Tuneable compromise |

### Phase 3: Advanced Processors

| Priority | Method | Effort | Impact |
|----------|--------|--------|--------|
| 7 | RPCA denoising | 1-2 days | Coherent/incoherent separation |
| 8 | NLM in TF domain | 2 days | Repetitive structure preservation |
| 9 | TV regularization | 1 day | Edge-preserving smoothing |

---

## References

1. Donoho, D.L. & Johnstone, I.M. (1994). "Ideal spatial adaptation by wavelet shrinkage"
2. Donoho, D.L. & Johnstone, I.M. (1995). "Adapting to unknown smoothness via wavelet shrinkage"
3. Chang, S.G., Yu, B. & Vetterli, M. (2000). "Adaptive wavelet thresholding for image denoising"
4. Sendur, L. & Selesnick, I.W. (2002). "Bivariate shrinkage functions for wavelet-based denoising"
5. Buades, A., Coll, B. & Morel, J.M. (2005). "A non-local algorithm for image denoising"
6. Candès, E.J. et al. (2011). "Robust principal component analysis"
7. Rudin, L.I., Osher, S. & Fatemi, E. (1992). "Total variation based image restoration"
8. Zhang, K. et al. (2017). "Beyond a Gaussian denoiser: Residual learning of deep CNN"
