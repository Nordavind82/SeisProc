/**
 * vDSP FFT Wrapper
 *
 * High-performance FFT using Apple's Accelerate framework (vDSP).
 * vDSP is highly optimized for Apple Silicon and outperforms custom
 * Metal shaders for most FFT sizes due to lower overhead.
 *
 * This is used in our hybrid approach:
 * - vDSP for FFT operations (highly optimized)
 * - Metal GPU for parallel operations (MAD, thresholding, masking)
 */

#ifndef SEISMIC_METAL_VDSP_FFT_H
#define SEISMIC_METAL_VDSP_FFT_H

#include <vector>
#include <complex>

namespace seismic_metal {

/**
 * 1D FFT (real-to-complex) using vDSP.
 *
 * @param input Real input signal
 * @param n Length of input (must be power of 2)
 * @return Complex spectrum (n/2 + 1 complex values)
 */
std::vector<std::complex<float>> vdsp_rfft(
    const float* input,
    int n
);

/**
 * 1D IFFT (complex-to-real) using vDSP.
 *
 * @param spectrum Complex spectrum
 * @param n_out Expected output length
 * @return Real signal
 */
std::vector<float> vdsp_irfft(
    const std::complex<float>* spectrum,
    int n_spectrum,
    int n_out
);

/**
 * Batch 1D FFT for multiple traces.
 *
 * @param traces Input traces [n_samples, n_traces] row-major
 * @param n_samples Number of samples per trace
 * @param n_traces Number of traces
 * @return Complex spectra [n_freqs, n_traces] where n_freqs = n_samples/2 + 1
 */
std::vector<std::complex<float>> vdsp_rfft_batch(
    const float* traces,
    int n_samples,
    int n_traces
);

/**
 * Batch 1D IFFT for multiple traces.
 *
 * @param spectra Complex spectra [n_freqs, n_traces]
 * @param n_freqs Number of frequency bins
 * @param n_traces Number of traces
 * @param n_out Output length per trace
 * @return Real traces [n_out, n_traces]
 */
std::vector<float> vdsp_irfft_batch(
    const std::complex<float>* spectra,
    int n_freqs,
    int n_traces,
    int n_out
);

/**
 * 2D FFT (real-to-complex) using vDSP.
 *
 * @param input Real 2D input [rows, cols]
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Complex spectrum
 */
std::vector<std::complex<float>> vdsp_rfft2d(
    const float* input,
    int rows,
    int cols
);

/**
 * 2D IFFT (complex-to-real) using vDSP.
 */
std::vector<float> vdsp_irfft2d(
    const std::complex<float>* spectrum,
    int rows,
    int cols
);

/**
 * 3D FFT for FKK filtering.
 *
 * Performs rfft along time axis, then fft along spatial axes.
 *
 * @param volume Input volume [nt, nx, ny]
 * @param nt, nx, ny Dimensions
 * @return Complex spectrum [nf, nx, ny] where nf = nt/2 + 1
 */
std::vector<std::complex<float>> vdsp_rfft3d(
    const float* volume,
    int nt,
    int nx,
    int ny
);

/**
 * 3D IFFT for FKK filtering.
 */
std::vector<float> vdsp_irfft3d(
    const std::complex<float>* spectrum,
    int nf,
    int nx,
    int ny,
    int nt_out
);

/**
 * Apply fftshift to 2D complex array (in-place).
 * Swaps quadrants so DC is at center.
 */
void fftshift_2d(
    std::complex<float>* data,
    int rows,
    int cols
);

/**
 * Apply ifftshift to 2D complex array (in-place).
 */
void ifftshift_2d(
    std::complex<float>* data,
    int rows,
    int cols
);

/**
 * Get next power of 2 >= n.
 */
int next_power_of_2(int n);

/**
 * Check if n is a power of 2.
 */
bool is_power_of_2(int n);

} // namespace seismic_metal

#endif // SEISMIC_METAL_VDSP_FFT_H
