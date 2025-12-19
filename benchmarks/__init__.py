"""
SeisProc Benchmarking Suite

Tools for profiling processor performance and identifying
bottlenecks for C++ kernel acceleration.
"""

from .synthetic_data_generator import (
    CrossspreadGeometry,
    generate_benchmark_dataset,
    load_benchmark_gather,
    load_benchmark_metadata,
)

from .profile_processors import (
    profile_dwt_denoise,
    profile_stft_denoise,
    profile_gabor_denoise,
    profile_fkk_filter,
    run_full_benchmark,
    ProfileResult,
    TimingResult,
)

__all__ = [
    'CrossspreadGeometry',
    'generate_benchmark_dataset',
    'load_benchmark_gather',
    'load_benchmark_metadata',
    'profile_dwt_denoise',
    'profile_stft_denoise',
    'profile_gabor_denoise',
    'profile_fkk_filter',
    'run_full_benchmark',
    'ProfileResult',
    'TimingResult',
]
