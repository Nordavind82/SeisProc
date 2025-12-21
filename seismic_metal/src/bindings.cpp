/**
 * Python Bindings for Seismic Metal Kernels
 *
 * Uses pybind11 for zero-copy NumPy array passing.
 * Exposes DWT, SWT, STFT, Gabor, and FKK kernels to Python.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "device_manager.h"
#include "dwt_kernel.h"
#include "stft_kernel.h"
#include "fkk_kernel.h"

namespace py = pybind11;

namespace seismic_metal {

// Device management wrappers
py::dict get_device_info_py() {
    py::dict info;
    info["available"] = is_available();
    info["device_name"] = get_device_name();
    return info;
}

// DWT wrapper - accepts NumPy array, returns NumPy array
py::tuple dwt_denoise_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> traces,
    const std::string& wavelet,
    int level,
    const std::string& threshold_mode,
    float threshold_k
) {
    py::buffer_info buf = traces.request();

    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be 2D array [n_samples, n_traces]");
    }

    int n_samples = buf.shape[0];
    int n_traces = buf.shape[1];
    const float* data_ptr = static_cast<float*>(buf.ptr);

    auto [output, metrics] = dwt_denoise(
        data_ptr, n_samples, n_traces,
        wavelet, level, threshold_k, threshold_mode
    );

    // Create output array
    py::array_t<float> result({n_samples, n_traces});
    float* result_ptr = static_cast<float*>(result.request().ptr);
    std::copy(output.begin(), output.end(), result_ptr);

    // Create metrics dict
    py::dict metrics_dict;
    metrics_dict["kernel_time_ms"] = metrics.kernel_time_ms;
    metrics_dict["total_time_ms"] = metrics.total_time_ms;
    metrics_dict["traces_processed"] = metrics.traces_processed;
    metrics_dict["samples_processed"] = metrics.samples_processed;

    return py::make_tuple(result, metrics_dict);
}

// SWT wrapper
py::tuple swt_denoise_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> traces,
    const std::string& wavelet,
    int level,
    const std::string& threshold_mode,
    float threshold_k
) {
    py::buffer_info buf = traces.request();

    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be 2D array [n_samples, n_traces]");
    }

    int n_samples = buf.shape[0];
    int n_traces = buf.shape[1];
    const float* data_ptr = static_cast<float*>(buf.ptr);

    auto [output, metrics] = swt_denoise(
        data_ptr, n_samples, n_traces,
        wavelet, level, threshold_k, threshold_mode
    );

    py::array_t<float> result({n_samples, n_traces});
    float* result_ptr = static_cast<float*>(result.request().ptr);
    std::copy(output.begin(), output.end(), result_ptr);

    py::dict metrics_dict;
    metrics_dict["kernel_time_ms"] = metrics.kernel_time_ms;
    metrics_dict["total_time_ms"] = metrics.total_time_ms;
    metrics_dict["traces_processed"] = metrics.traces_processed;
    metrics_dict["samples_processed"] = metrics.samples_processed;

    return py::make_tuple(result, metrics_dict);
}

// STFT wrapper
py::tuple stft_denoise_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> traces,
    int nperseg,
    int noverlap,
    int aperture,
    float threshold_k,
    float fmin,
    float fmax,
    float sample_rate,
    bool low_amp_protection,
    float low_amp_factor
) {
    py::buffer_info buf = traces.request();

    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be 2D array [n_samples, n_traces]");
    }

    int n_samples = buf.shape[0];
    int n_traces = buf.shape[1];
    const float* data_ptr = static_cast<float*>(buf.ptr);

    auto [output, metrics] = stft_denoise(
        data_ptr, n_samples, n_traces,
        nperseg, noverlap, aperture,
        threshold_k, fmin, fmax, sample_rate,
        low_amp_protection, low_amp_factor
    );

    py::array_t<float> result({n_samples, n_traces});
    float* result_ptr = static_cast<float*>(result.request().ptr);
    std::copy(output.begin(), output.end(), result_ptr);

    py::dict metrics_dict;
    metrics_dict["kernel_time_ms"] = metrics.kernel_time_ms;
    metrics_dict["total_time_ms"] = metrics.total_time_ms;
    metrics_dict["traces_processed"] = metrics.traces_processed;
    metrics_dict["samples_processed"] = metrics.samples_processed;

    return py::make_tuple(result, metrics_dict);
}

// Gabor wrapper
py::tuple gabor_denoise_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> traces,
    int window_size,
    float sigma,
    float overlap_pct,
    int aperture,
    float threshold_k,
    float fmin,
    float fmax,
    float sample_rate
) {
    py::buffer_info buf = traces.request();

    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be 2D array [n_samples, n_traces]");
    }

    int n_samples = buf.shape[0];
    int n_traces = buf.shape[1];
    const float* data_ptr = static_cast<float*>(buf.ptr);

    auto [output, metrics] = gabor_denoise(
        data_ptr, n_samples, n_traces,
        window_size, sigma, overlap_pct,
        aperture, threshold_k, fmin, fmax, sample_rate
    );

    py::array_t<float> result({n_samples, n_traces});
    float* result_ptr = static_cast<float*>(result.request().ptr);
    std::copy(output.begin(), output.end(), result_ptr);

    py::dict metrics_dict;
    metrics_dict["kernel_time_ms"] = metrics.kernel_time_ms;
    metrics_dict["total_time_ms"] = metrics.total_time_ms;
    metrics_dict["traces_processed"] = metrics.traces_processed;
    metrics_dict["samples_processed"] = metrics.samples_processed;

    return py::make_tuple(result, metrics_dict);
}

// FKK wrapper - accepts 3D volume
py::tuple fkk_filter_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> volume,
    float dt,
    float dx,
    float dy,
    float v_min,
    float v_max,
    const std::string& mode,
    bool preserve_dc
) {
    py::buffer_info buf = volume.request();

    if (buf.ndim != 3) {
        throw std::runtime_error("Input must be 3D array [nt, nx, ny]");
    }

    int nt = buf.shape[0];
    int nx = buf.shape[1];
    int ny = buf.shape[2];
    const float* data_ptr = static_cast<float*>(buf.ptr);

    auto [output, metrics] = fkk_filter(
        data_ptr, nt, nx, ny,
        dt, dx, dy, v_min, v_max, mode, preserve_dc
    );

    py::array_t<float> result({nt, nx, ny});
    float* result_ptr = static_cast<float*>(result.request().ptr);
    std::copy(output.begin(), output.end(), result_ptr);

    py::dict metrics_dict;
    metrics_dict["kernel_time_ms"] = metrics.kernel_time_ms;
    metrics_dict["total_time_ms"] = metrics.total_time_ms;
    metrics_dict["traces_processed"] = metrics.traces_processed;
    metrics_dict["samples_processed"] = metrics.samples_processed;

    return py::make_tuple(result, metrics_dict);
}

} // namespace seismic_metal

PYBIND11_MODULE(seismic_metal, m) {
    m.doc() = "Seismic Metal Kernels - GPU-accelerated seismic processing for Apple Silicon";

    // Device management
    m.def("initialize", &seismic_metal::initialize_device,
          py::arg("shader_path") = "",
          "Initialize Metal device and load shaders");

    m.def("is_available", &seismic_metal::is_available,
          "Check if Metal GPU acceleration is available");

    m.def("get_device_info", &seismic_metal::get_device_info_py,
          "Get Metal device information");

    m.def("cleanup", &seismic_metal::cleanup,
          "Release Metal resources");

    // DWT denoising
    m.def("dwt_denoise", &seismic_metal::dwt_denoise_py,
          py::arg("traces"),
          py::arg("wavelet") = "db4",
          py::arg("level") = 4,
          py::arg("threshold_mode") = "soft",
          py::arg("threshold_k") = 3.0f,
          R"doc(
          Apply DWT denoising using Metal GPU acceleration.

          Parameters
          ----------
          traces : ndarray
              Input traces [n_samples, n_traces]
          wavelet : str
              Wavelet name ('db4', 'sym4')
          level : int
              Decomposition level
          threshold_mode : str
              'soft' or 'hard' thresholding
          threshold_k : float
              Threshold multiplier for MAD

          Returns
          -------
          tuple
              (denoised_traces, metrics_dict)
          )doc");

    // SWT denoising
    m.def("swt_denoise", &seismic_metal::swt_denoise_py,
          py::arg("traces"),
          py::arg("wavelet") = "db4",
          py::arg("level") = 4,
          py::arg("threshold_mode") = "soft",
          py::arg("threshold_k") = 3.0f,
          R"doc(
          Apply SWT (Stationary Wavelet Transform) denoising using Metal GPU acceleration.

          Parameters
          ----------
          traces : ndarray
              Input traces [n_samples, n_traces]
          wavelet : str
              Wavelet name ('db4', 'sym4')
          level : int
              Decomposition level
          threshold_mode : str
              'soft' or 'hard' thresholding
          threshold_k : float
              Threshold multiplier for MAD

          Returns
          -------
          tuple
              (denoised_traces, metrics_dict)
          )doc");

    // STFT denoising
    m.def("stft_denoise", &seismic_metal::stft_denoise_py,
          py::arg("traces"),
          py::arg("nperseg") = 64,
          py::arg("noverlap") = 48,
          py::arg("aperture") = 21,
          py::arg("threshold_k") = 3.0f,
          py::arg("fmin") = 0.0f,
          py::arg("fmax") = 0.0f,
          py::arg("sample_rate") = 500.0f,
          py::arg("low_amp_protection") = true,
          py::arg("low_amp_factor") = 0.3f,
          R"doc(
          Apply STFT denoising using Metal GPU acceleration.

          Parameters
          ----------
          traces : ndarray
              Input traces [n_samples, n_traces]
          nperseg : int
              FFT window size
          noverlap : int
              Overlap between windows
          aperture : int
              Spatial aperture for median computation
          threshold_k : float
              Threshold multiplier for MAD
          fmin, fmax : float
              Frequency range (0 = no limit)
          sample_rate : float
              Sample rate in Hz
          low_amp_protection : bool
              Prevent inflation of low-amplitude samples (default True)
          low_amp_factor : float
              Threshold for low-amplitude protection as fraction of median (default 0.3)

          Returns
          -------
          tuple
              (denoised_traces, metrics_dict)
          )doc");

    // Gabor denoising
    m.def("gabor_denoise", &seismic_metal::gabor_denoise_py,
          py::arg("traces"),
          py::arg("window_size") = 64,
          py::arg("sigma") = 0.0f,
          py::arg("overlap_pct") = 75.0f,
          py::arg("aperture") = 21,
          py::arg("threshold_k") = 3.0f,
          py::arg("fmin") = 0.0f,
          py::arg("fmax") = 0.0f,
          py::arg("sample_rate") = 500.0f,
          R"doc(
          Apply Gabor denoising using Metal GPU acceleration.

          Parameters
          ----------
          traces : ndarray
              Input traces [n_samples, n_traces]
          window_size : int
              Gaussian window size
          sigma : float
              Gaussian sigma (0 = auto)
          overlap_pct : float
              Overlap percentage
          aperture : int
              Spatial aperture for median computation
          threshold_k : float
              Threshold multiplier for MAD
          fmin, fmax : float
              Frequency range (0 = no limit)
          sample_rate : float
              Sample rate in Hz

          Returns
          -------
          tuple
              (denoised_traces, metrics_dict)
          )doc");

    // FKK filtering
    m.def("fkk_filter", &seismic_metal::fkk_filter_py,
          py::arg("volume"),
          py::arg("dt"),
          py::arg("dx"),
          py::arg("dy"),
          py::arg("v_min"),
          py::arg("v_max"),
          py::arg("mode") = "reject",
          py::arg("preserve_dc") = true,
          R"doc(
          Apply 3D FK filtering using Metal GPU acceleration.

          Parameters
          ----------
          volume : ndarray
              Input volume [nt, nx, ny]
          dt : float
              Time sample interval (seconds)
          dx : float
              Inline spacing (meters)
          dy : float
              Crossline spacing (meters)
          v_min : float
              Minimum velocity for filter (m/s)
          v_max : float
              Maximum velocity for filter (m/s)
          mode : str
              'reject' to remove velocities in range, 'pass' to keep only
          preserve_dc : bool
              Preserve DC component

          Returns
          -------
          tuple
              (filtered_volume, metrics_dict)
          )doc");

    // Version info
    m.attr("__version__") = "1.0.0";
}
