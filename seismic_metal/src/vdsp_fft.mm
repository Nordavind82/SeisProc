/**
 * vDSP FFT Implementation
 *
 * Uses Apple's Accelerate framework for high-performance FFT operations.
 * vDSP is highly optimized for Apple Silicon and provides better performance
 * than custom GPU implementations for most FFT sizes.
 */

#import <Accelerate/Accelerate.h>
#include "vdsp_fft.h"
#include <cmath>
#include <algorithm>
#include <thread>

namespace seismic_metal {

// Thread-local FFT setup cache for performance
static thread_local FFTSetup g_fft_setup = nullptr;
static thread_local int g_fft_log2n = 0;

static FFTSetup get_fft_setup(int log2n) {
    if (g_fft_setup == nullptr || g_fft_log2n < log2n) {
        if (g_fft_setup != nullptr) {
            vDSP_destroy_fftsetup(g_fft_setup);
        }
        g_fft_setup = vDSP_create_fftsetup(log2n, FFT_RADIX2);
        g_fft_log2n = log2n;
    }
    return g_fft_setup;
}

int next_power_of_2(int n) {
    if (n <= 1) return 1;
    int p = 1;
    while (p < n) p *= 2;
    return p;
}

bool is_power_of_2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

std::vector<std::complex<float>> vdsp_rfft(const float* input, int n) {
    // Ensure power of 2
    int n_padded = next_power_of_2(n);
    int log2n = (int)log2(n_padded);

    // Get FFT setup
    FFTSetup setup = get_fft_setup(log2n);

    // Prepare split complex format for vDSP
    std::vector<float> real(n_padded / 2);
    std::vector<float> imag(n_padded / 2);

    DSPSplitComplex split = {real.data(), imag.data()};

    // Copy input and convert to split complex (pack even/odd samples)
    std::vector<float> padded(n_padded, 0.0f);
    std::copy(input, input + n, padded.begin());

    // Convert real signal to split complex format
    vDSP_ctoz((DSPComplex*)padded.data(), 2, &split, 1, n_padded / 2);

    // Perform FFT
    vDSP_fft_zrip(setup, &split, 1, log2n, FFT_FORWARD);

    // Scale (vDSP doesn't normalize)
    float scale = 0.5f;
    vDSP_vsmul(split.realp, 1, &scale, split.realp, 1, n_padded / 2);
    vDSP_vsmul(split.imagp, 1, &scale, split.imagp, 1, n_padded / 2);

    // Convert to complex output
    // vDSP packs DC and Nyquist specially: split.realp[0] = DC, split.imagp[0] = Nyquist
    int n_freqs = n_padded / 2 + 1;
    std::vector<std::complex<float>> output(n_freqs);

    output[0] = std::complex<float>(split.realp[0], 0.0f);  // DC
    for (int i = 1; i < n_padded / 2; i++) {
        output[i] = std::complex<float>(split.realp[i], split.imagp[i]);
    }
    output[n_padded / 2] = std::complex<float>(split.imagp[0], 0.0f);  // Nyquist

    return output;
}

std::vector<float> vdsp_irfft(
    const std::complex<float>* spectrum,
    int n_spectrum,
    int n_out
) {
    int n_padded = (n_spectrum - 1) * 2;
    int log2n = (int)log2(n_padded);

    FFTSetup setup = get_fft_setup(log2n);

    // Prepare split complex format
    std::vector<float> real(n_padded / 2);
    std::vector<float> imag(n_padded / 2);

    // Unpack: DC in realp[0], Nyquist in imagp[0]
    real[0] = spectrum[0].real();
    imag[0] = spectrum[n_spectrum - 1].real();  // Nyquist

    for (int i = 1; i < n_padded / 2; i++) {
        real[i] = spectrum[i].real();
        imag[i] = spectrum[i].imag();
    }

    DSPSplitComplex split = {real.data(), imag.data()};

    // Perform inverse FFT
    vDSP_fft_zrip(setup, &split, 1, log2n, FFT_INVERSE);

    // Convert back to interleaved real format
    std::vector<float> output(n_padded);
    vDSP_ztoc(&split, 1, (DSPComplex*)output.data(), 2, n_padded / 2);

    // Scale by 1/n
    float scale = 1.0f / n_padded;
    vDSP_vsmul(output.data(), 1, &scale, output.data(), 1, n_padded);

    // Return requested length
    output.resize(n_out);
    return output;
}

std::vector<std::complex<float>> vdsp_rfft_batch(
    const float* traces,
    int n_samples,
    int n_traces
) {
    int n_padded = next_power_of_2(n_samples);
    int n_freqs = n_padded / 2 + 1;

    std::vector<std::complex<float>> output(n_freqs * n_traces);

    // Process traces in parallel using thread pool
    int n_threads = std::min((int)std::thread::hardware_concurrency(), n_traces);
    std::vector<std::thread> threads;

    auto process_range = [&](int start, int end) {
        for (int t = start; t < end; t++) {
            auto spectrum = vdsp_rfft(traces + t * n_samples, n_samples);
            for (int f = 0; f < n_freqs; f++) {
                output[f * n_traces + t] = spectrum[f];
            }
        }
    };

    int chunk_size = (n_traces + n_threads - 1) / n_threads;
    for (int i = 0; i < n_threads; i++) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, n_traces);
        if (start < end) {
            threads.emplace_back(process_range, start, end);
        }
    }

    for (auto& t : threads) {
        t.join();
    }

    return output;
}

std::vector<float> vdsp_irfft_batch(
    const std::complex<float>* spectra,
    int n_freqs,
    int n_traces,
    int n_out
) {
    std::vector<float> output(n_out * n_traces);

    int n_threads = std::min((int)std::thread::hardware_concurrency(), n_traces);
    std::vector<std::thread> threads;

    auto process_range = [&](int start, int end) {
        std::vector<std::complex<float>> spectrum(n_freqs);
        for (int t = start; t < end; t++) {
            // Gather spectrum for this trace
            for (int f = 0; f < n_freqs; f++) {
                spectrum[f] = spectra[f * n_traces + t];
            }
            auto trace = vdsp_irfft(spectrum.data(), n_freqs, n_out);
            std::copy(trace.begin(), trace.end(), output.data() + t * n_out);
        }
    };

    int chunk_size = (n_traces + n_threads - 1) / n_threads;
    for (int i = 0; i < n_threads; i++) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, n_traces);
        if (start < end) {
            threads.emplace_back(process_range, start, end);
        }
    }

    for (auto& t : threads) {
        t.join();
    }

    return output;
}

std::vector<std::complex<float>> vdsp_rfft2d(
    const float* input,
    int rows,
    int cols
) {
    // 2D FFT: FFT along columns first, then rows
    int cols_padded = next_power_of_2(cols);
    int rows_padded = next_power_of_2(rows);
    int n_freq_cols = cols_padded / 2 + 1;

    // Step 1: FFT along each row (real-to-complex)
    std::vector<std::complex<float>> row_fft(rows * n_freq_cols);
    for (int r = 0; r < rows; r++) {
        auto spectrum = vdsp_rfft(input + r * cols, cols);
        for (int c = 0; c < n_freq_cols; c++) {
            row_fft[r * n_freq_cols + c] = spectrum[c];
        }
    }

    // Step 2: FFT along each column (complex-to-complex)
    int log2rows = (int)log2(rows_padded);
    FFTSetup setup = get_fft_setup(log2rows);

    std::vector<std::complex<float>> output(rows_padded * n_freq_cols);

    for (int c = 0; c < n_freq_cols; c++) {
        // Extract column
        std::vector<float> real(rows_padded, 0.0f);
        std::vector<float> imag(rows_padded, 0.0f);

        for (int r = 0; r < rows; r++) {
            real[r] = row_fft[r * n_freq_cols + c].real();
            imag[r] = row_fft[r * n_freq_cols + c].imag();
        }

        DSPSplitComplex split = {real.data(), imag.data()};

        // Complex FFT
        vDSP_fft_zip(setup, &split, 1, log2rows, FFT_FORWARD);

        // Store result
        for (int r = 0; r < rows_padded; r++) {
            output[r * n_freq_cols + c] = std::complex<float>(real[r], imag[r]);
        }
    }

    return output;
}

std::vector<float> vdsp_irfft2d(
    const std::complex<float>* spectrum,
    int rows,
    int cols
) {
    int n_freq_cols = cols / 2 + 1;
    int log2rows = (int)log2(rows);
    FFTSetup setup = get_fft_setup(log2rows);

    // Step 1: IFFT along each column (complex-to-complex)
    std::vector<std::complex<float>> col_ifft(rows * n_freq_cols);

    for (int c = 0; c < n_freq_cols; c++) {
        std::vector<float> real(rows);
        std::vector<float> imag(rows);

        for (int r = 0; r < rows; r++) {
            real[r] = spectrum[r * n_freq_cols + c].real();
            imag[r] = spectrum[r * n_freq_cols + c].imag();
        }

        DSPSplitComplex split = {real.data(), imag.data()};
        vDSP_fft_zip(setup, &split, 1, log2rows, FFT_INVERSE);

        float scale = 1.0f / rows;
        vDSP_vsmul(real.data(), 1, &scale, real.data(), 1, rows);
        vDSP_vsmul(imag.data(), 1, &scale, imag.data(), 1, rows);

        for (int r = 0; r < rows; r++) {
            col_ifft[r * n_freq_cols + c] = std::complex<float>(real[r], imag[r]);
        }
    }

    // Step 2: IRFFT along each row
    std::vector<float> output(rows * cols);
    for (int r = 0; r < rows; r++) {
        auto row = vdsp_irfft(col_ifft.data() + r * n_freq_cols, n_freq_cols, cols);
        std::copy(row.begin(), row.end(), output.data() + r * cols);
    }

    return output;
}

std::vector<std::complex<float>> vdsp_rfft3d(
    const float* volume,
    int nt,
    int nx,
    int ny
) {
    int nt_padded = next_power_of_2(nt);
    int nx_padded = next_power_of_2(nx);
    int ny_padded = next_power_of_2(ny);
    int nf = nt_padded / 2 + 1;

    // Step 1: RFFT along time axis for each spatial position
    std::vector<std::complex<float>> time_fft(nf * nx * ny);

    int n_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    auto process_time_fft = [&](int start_x, int end_x) {
        for (int x = start_x; x < end_x; x++) {
            for (int y = 0; y < ny; y++) {
                // Extract time series for this spatial position
                std::vector<float> trace(nt);
                for (int t = 0; t < nt; t++) {
                    trace[t] = volume[t * nx * ny + x * ny + y];
                }

                auto spectrum = vdsp_rfft(trace.data(), nt);

                for (int f = 0; f < nf; f++) {
                    time_fft[f * nx * ny + x * ny + y] = spectrum[f];
                }
            }
        }
    };

    int chunk_size = (nx + n_threads - 1) / n_threads;
    for (int i = 0; i < (int)n_threads; i++) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, nx);
        if (start < end) {
            threads.emplace_back(process_time_fft, start, end);
        }
    }
    for (auto& t : threads) t.join();
    threads.clear();

    // Step 2: Complex FFT along X axis for each frequency and Y
    int log2nx = (int)log2(nx_padded);

    std::vector<std::complex<float>> x_fft(nf * nx_padded * ny);

    auto process_x_fft = [&](int start_f, int end_f) {
        FFTSetup setup = vDSP_create_fftsetup(log2nx, FFT_RADIX2);

        for (int f = start_f; f < end_f; f++) {
            for (int y = 0; y < ny; y++) {
                std::vector<float> real(nx_padded, 0.0f);
                std::vector<float> imag(nx_padded, 0.0f);

                for (int x = 0; x < nx; x++) {
                    real[x] = time_fft[f * nx * ny + x * ny + y].real();
                    imag[x] = time_fft[f * nx * ny + x * ny + y].imag();
                }

                DSPSplitComplex split = {real.data(), imag.data()};
                vDSP_fft_zip(setup, &split, 1, log2nx, FFT_FORWARD);

                for (int x = 0; x < nx_padded; x++) {
                    x_fft[f * nx_padded * ny + x * ny + y] =
                        std::complex<float>(real[x], imag[x]);
                }
            }
        }

        vDSP_destroy_fftsetup(setup);
    };

    chunk_size = (nf + n_threads - 1) / n_threads;
    for (int i = 0; i < (int)n_threads; i++) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, nf);
        if (start < end) {
            threads.emplace_back(process_x_fft, start, end);
        }
    }
    for (auto& t : threads) t.join();
    threads.clear();

    // Step 3: Complex FFT along Y axis for each frequency and X
    int log2ny = (int)log2(ny_padded);

    std::vector<std::complex<float>> output(nf * nx_padded * ny_padded);

    auto process_y_fft = [&](int start_f, int end_f) {
        FFTSetup setup = vDSP_create_fftsetup(log2ny, FFT_RADIX2);

        for (int f = start_f; f < end_f; f++) {
            for (int x = 0; x < nx_padded; x++) {
                std::vector<float> real(ny_padded, 0.0f);
                std::vector<float> imag(ny_padded, 0.0f);

                for (int y = 0; y < ny; y++) {
                    real[y] = x_fft[f * nx_padded * ny + x * ny + y].real();
                    imag[y] = x_fft[f * nx_padded * ny + x * ny + y].imag();
                }

                DSPSplitComplex split = {real.data(), imag.data()};
                vDSP_fft_zip(setup, &split, 1, log2ny, FFT_FORWARD);

                for (int y = 0; y < ny_padded; y++) {
                    output[f * nx_padded * ny_padded + x * ny_padded + y] =
                        std::complex<float>(real[y], imag[y]);
                }
            }
        }

        vDSP_destroy_fftsetup(setup);
    };

    chunk_size = (nf + n_threads - 1) / n_threads;
    for (int i = 0; i < (int)n_threads; i++) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, nf);
        if (start < end) {
            threads.emplace_back(process_y_fft, start, end);
        }
    }
    for (auto& t : threads) t.join();

    return output;
}

std::vector<float> vdsp_irfft3d(
    const std::complex<float>* spectrum,
    int nf,
    int nx,
    int ny,
    int nt_out
) {
    int log2nx = (int)log2(nx);
    int log2ny = (int)log2(ny);

    int n_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    // Step 1: IFFT along Y axis
    std::vector<std::complex<float>> y_ifft(nf * nx * ny);

    auto process_y_ifft = [&](int start_f, int end_f) {
        FFTSetup setup = vDSP_create_fftsetup(log2ny, FFT_RADIX2);

        for (int f = start_f; f < end_f; f++) {
            for (int x = 0; x < nx; x++) {
                std::vector<float> real(ny);
                std::vector<float> imag(ny);

                for (int y = 0; y < ny; y++) {
                    real[y] = spectrum[f * nx * ny + x * ny + y].real();
                    imag[y] = spectrum[f * nx * ny + x * ny + y].imag();
                }

                DSPSplitComplex split = {real.data(), imag.data()};
                vDSP_fft_zip(setup, &split, 1, log2ny, FFT_INVERSE);

                float scale = 1.0f / ny;
                vDSP_vsmul(real.data(), 1, &scale, real.data(), 1, ny);
                vDSP_vsmul(imag.data(), 1, &scale, imag.data(), 1, ny);

                for (int y = 0; y < ny; y++) {
                    y_ifft[f * nx * ny + x * ny + y] =
                        std::complex<float>(real[y], imag[y]);
                }
            }
        }

        vDSP_destroy_fftsetup(setup);
    };

    int chunk_size = (nf + n_threads - 1) / n_threads;
    for (int i = 0; i < (int)n_threads; i++) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, nf);
        if (start < end) {
            threads.emplace_back(process_y_ifft, start, end);
        }
    }
    for (auto& t : threads) t.join();
    threads.clear();

    // Step 2: IFFT along X axis
    std::vector<std::complex<float>> x_ifft(nf * nx * ny);

    auto process_x_ifft = [&](int start_f, int end_f) {
        FFTSetup setup = vDSP_create_fftsetup(log2nx, FFT_RADIX2);

        for (int f = start_f; f < end_f; f++) {
            for (int y = 0; y < ny; y++) {
                std::vector<float> real(nx);
                std::vector<float> imag(nx);

                for (int x = 0; x < nx; x++) {
                    real[x] = y_ifft[f * nx * ny + x * ny + y].real();
                    imag[x] = y_ifft[f * nx * ny + x * ny + y].imag();
                }

                DSPSplitComplex split = {real.data(), imag.data()};
                vDSP_fft_zip(setup, &split, 1, log2nx, FFT_INVERSE);

                float scale = 1.0f / nx;
                vDSP_vsmul(real.data(), 1, &scale, real.data(), 1, nx);
                vDSP_vsmul(imag.data(), 1, &scale, imag.data(), 1, nx);

                for (int x = 0; x < nx; x++) {
                    x_ifft[f * nx * ny + x * ny + y] =
                        std::complex<float>(real[x], imag[x]);
                }
            }
        }

        vDSP_destroy_fftsetup(setup);
    };

    chunk_size = (nf + n_threads - 1) / n_threads;
    for (int i = 0; i < (int)n_threads; i++) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, nf);
        if (start < end) {
            threads.emplace_back(process_x_ifft, start, end);
        }
    }
    for (auto& t : threads) t.join();
    threads.clear();

    // Step 3: IRFFT along time axis
    int nx_orig = nx;  // We need original dimensions
    int ny_orig = ny;

    std::vector<float> output(nt_out * nx_orig * ny_orig);

    auto process_time_ifft = [&](int start_x, int end_x) {
        for (int x = start_x; x < end_x; x++) {
            for (int y = 0; y < ny_orig; y++) {
                std::vector<std::complex<float>> freq_trace(nf);
                for (int f = 0; f < nf; f++) {
                    freq_trace[f] = x_ifft[f * nx * ny + x * ny + y];
                }

                auto trace = vdsp_irfft(freq_trace.data(), nf, nt_out);

                for (int t = 0; t < nt_out; t++) {
                    output[t * nx_orig * ny_orig + x * ny_orig + y] = trace[t];
                }
            }
        }
    };

    chunk_size = (nx_orig + n_threads - 1) / n_threads;
    for (int i = 0; i < (int)n_threads; i++) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, nx_orig);
        if (start < end) {
            threads.emplace_back(process_time_ifft, start, end);
        }
    }
    for (auto& t : threads) t.join();

    return output;
}

void fftshift_2d(std::complex<float>* data, int rows, int cols) {
    int half_rows = rows / 2;
    int half_cols = cols / 2;

    // Swap quadrants
    for (int r = 0; r < half_rows; r++) {
        for (int c = 0; c < half_cols; c++) {
            // Swap (r, c) with (r + half_rows, c + half_cols)
            std::swap(data[r * cols + c],
                     data[(r + half_rows) * cols + (c + half_cols)]);
            // Swap (r, c + half_cols) with (r + half_rows, c)
            std::swap(data[r * cols + (c + half_cols)],
                     data[(r + half_rows) * cols + c]);
        }
    }
}

void ifftshift_2d(std::complex<float>* data, int rows, int cols) {
    // For even dimensions, ifftshift == fftshift
    // For odd dimensions, they differ slightly
    fftshift_2d(data, rows, cols);
}

} // namespace seismic_metal
