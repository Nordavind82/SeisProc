#!/usr/bin/env python3
"""
Resource Monitor for SeisProc

Monitors CPU, GPU (Metal/MPS), memory, and disk I/O usage.
Run in a separate terminal while executing migration jobs.

Usage:
    python tools/resource_monitor.py [--interval 5] [--output monitor.log]
"""

import argparse
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path


def get_cpu_usage():
    """Get CPU usage percentage."""
    try:
        # Use top for CPU usage on macOS
        result = subprocess.run(
            ['top', '-l', '1', '-n', '0', '-stats', 'cpu'],
            capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if 'CPU usage' in line:
                # Parse: CPU usage: 12.5% user, 8.3% sys, 79.2% idle
                parts = line.split(',')
                user = float(parts[0].split(':')[1].strip().replace('% user', ''))
                sys_pct = float(parts[1].strip().replace('% sys', ''))
                return user + sys_pct, user, sys_pct
    except Exception as e:
        pass
    return 0.0, 0.0, 0.0


def get_memory_usage():
    """Get memory usage in GB."""
    try:
        result = subprocess.run(
            ['vm_stat'],
            capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().split('\n')
        stats = {}
        page_size = 16384  # Default for Apple Silicon

        for line in lines:
            if ':' in line:
                key, value = line.split(':')
                value = value.strip().replace('.', '')
                if value.isdigit():
                    stats[key.strip()] = int(value)

        # Calculate memory
        pages_active = stats.get('Pages active', 0)
        pages_wired = stats.get('Pages wired down', 0)
        pages_compressed = stats.get('Pages occupied by compressor', 0)
        pages_free = stats.get('Pages free', 0)

        used_bytes = (pages_active + pages_wired + pages_compressed) * page_size
        free_bytes = pages_free * page_size
        total_bytes = used_bytes + free_bytes

        return used_bytes / (1024**3), total_bytes / (1024**3)
    except Exception as e:
        pass
    return 0.0, 0.0


def get_gpu_stats_apple():
    """
    Get Apple Silicon GPU stats using sudo powermetrics.
    Returns dict with GPU usage percentage and power.
    """
    gpu_stats = {
        'gpu_usage': 0.0,
        'gpu_power': 0.0,
        'available': False,
        'method': 'none'
    }

    # Method 1: Try powermetrics (most accurate, requires sudo or special permissions)
    try:
        result = subprocess.run(
            ['sudo', '-n', 'powermetrics', '-n', '1', '-i', '100', '--samplers', 'gpu_power'],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'GPU Active' in line or 'GPU active' in line:
                    # Parse: GPU Active: 45%
                    try:
                        pct = float(line.split(':')[1].strip().replace('%', ''))
                        gpu_stats['gpu_usage'] = pct
                        gpu_stats['available'] = True
                        gpu_stats['method'] = 'powermetrics'
                    except:
                        pass
                elif 'GPU Power' in line:
                    try:
                        power = float(line.split(':')[1].strip().replace('mW', '').replace('W', ''))
                        gpu_stats['gpu_power'] = power
                    except:
                        pass
            if gpu_stats['available']:
                return gpu_stats
    except Exception:
        pass

    # Method 2: Use ioreg to check GPU activity
    try:
        result = subprocess.run(
            ['ioreg', '-r', '-c', 'IOGPUDevice'],
            capture_output=True, text=True, timeout=3
        )
        if 'IOGPUDevice' in result.stdout:
            gpu_stats['available'] = True
            gpu_stats['method'] = 'ioreg'
            # Can't get usage %, but can confirm GPU exists
    except Exception:
        pass

    # Method 3: Check Activity Monitor's GPU history via sysctl
    try:
        result = subprocess.run(
            ['sysctl', 'machdep.gpu'],
            capture_output=True, text=True, timeout=2
        )
        # This might not exist on all systems
    except Exception:
        pass

    return gpu_stats


def get_gpu_memory_pytorch():
    """Get PyTorch MPS memory usage and stats."""
    stats = {
        'available': False,
        'allocated_gb': 0.0,
        'device': 'none',
        'driver': 'none'
    }

    try:
        import torch

        if torch.backends.mps.is_available():
            stats['available'] = True
            stats['device'] = 'Apple MPS'
            stats['driver'] = 'Metal'

            # Try to get allocated memory (MPS has limited introspection)
            # Note: current_allocated_memory() only shows THIS process's allocations
            # For cross-process GPU monitoring, use driver_allocated_memory() or powermetrics
            try:
                # First try driver memory (shows all GPU allocations)
                driver_mem = torch.mps.driver_allocated_memory() / (1024**3)
                current_mem = torch.mps.current_allocated_memory() / (1024**3)
                # Use driver memory if it's higher (indicates other processes using GPU)
                stats['allocated_gb'] = max(driver_mem, current_mem)
                stats['driver_gb'] = driver_mem
            except AttributeError:
                try:
                    stats['allocated_gb'] = torch.mps.current_allocated_memory() / (1024**3)
                except:
                    pass

        elif torch.cuda.is_available():
            stats['available'] = True
            stats['device'] = torch.cuda.get_device_name(0)
            stats['driver'] = 'CUDA'
            stats['allocated_gb'] = torch.cuda.memory_allocated(0) / (1024**3)

    except ImportError:
        pass
    except Exception as e:
        stats['error'] = str(e)

    return stats


def get_metal_gpu_processes():
    """Get processes using Metal/GPU on macOS."""
    gpu_procs = []
    try:
        # Use ps to find processes, then check if they're using GPU
        result = subprocess.run(
            ['ps', '-eo', 'pid,pcpu,pmem,comm'],
            capture_output=True, text=True, timeout=5
        )

        # Known GPU-using processes
        gpu_indicators = ['python', 'Python', 'metal', 'Metal', 'WindowServer', 'kernel_task']

        for line in result.stdout.split('\n')[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 4:
                pid = parts[0]
                cpu = float(parts[1]) if parts[1].replace('.', '').isdigit() else 0
                mem = float(parts[2]) if parts[2].replace('.', '').isdigit() else 0
                cmd = parts[3]

                if any(ind.lower() in cmd.lower() for ind in gpu_indicators):
                    if cpu > 1.0 or mem > 1.0:  # Only show active processes
                        gpu_procs.append({
                            'pid': pid,
                            'cpu': cpu,
                            'mem': mem,
                            'cmd': cmd[:30]
                        })
    except Exception:
        pass

    return gpu_procs


def get_python_processes():
    """Get Python process memory and CPU usage."""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True, text=True, timeout=5
        )
        python_procs = []
        for line in result.stdout.split('\n'):
            if 'python' in line.lower() and 'resource_monitor' not in line:
                parts = line.split()
                if len(parts) >= 11:
                    cpu = float(parts[2])
                    mem = float(parts[3])
                    rss_kb = int(parts[5]) if parts[5].isdigit() else 0
                    cmd = ' '.join(parts[10:])[:50]
                    python_procs.append({
                        'cpu': cpu,
                        'mem': mem,
                        'rss_gb': rss_kb / (1024 * 1024),
                        'cmd': cmd
                    })
        return python_procs
    except Exception:
        pass
    return []


def get_disk_io():
    """Get disk I/O stats."""
    try:
        result = subprocess.run(
            ['iostat', '-d', '-c', '1'],
            capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().split('\n')
        # Parse iostat output
        for i, line in enumerate(lines):
            if 'disk' in line.lower() and i + 1 < len(lines):
                # Header line found
                data_line = lines[i + 1].split()
                if len(data_line) >= 3:
                    kb_per_t = float(data_line[0]) if data_line[0].replace('.', '').isdigit() else 0
                    tps = float(data_line[1]) if data_line[1].replace('.', '').isdigit() else 0
                    mb_per_s = float(data_line[2]) if data_line[2].replace('.', '').isdigit() else 0
                    return mb_per_s, tps
    except Exception:
        pass
    return 0.0, 0.0


def format_bar(value, max_val=100, width=20):
    """Create ASCII progress bar."""
    filled = int(width * min(value, max_val) / max_val)
    return '█' * filled + '░' * (width - filled)


def monitor_loop(interval=5, output_file=None, duration=None, show_gpu=True):
    """Main monitoring loop."""
    print("=" * 80)
    print("SeisProc Resource Monitor")
    print("=" * 80)
    print(f"Interval: {interval}s | GPU monitoring: {show_gpu}")
    print(f"Output: {output_file if output_file else 'console only'}")
    print(f"Press Ctrl+C to stop")
    print("=" * 80)

    # Check GPU availability at startup
    if show_gpu:
        gpu_pytorch = get_gpu_memory_pytorch()
        gpu_apple = get_gpu_stats_apple()
        print(f"\nGPU Status:")
        print(f"  PyTorch: {gpu_pytorch['device']} ({gpu_pytorch['driver']})")
        print(f"  Apple GPU: {gpu_apple['method']}")
        if gpu_apple['method'] == 'powermetrics':
            print(f"  (Full GPU stats available via powermetrics)")
        else:
            print(f"  (For detailed GPU stats, run: sudo powermetrics --samplers gpu_power)")

    print("=" * 80)
    print()

    start_time = time.time()
    log_lines = []

    # Header - compact version
    if show_gpu:
        header = (
            "Timestamp            | CPU%  | Mem GB | PyRSS GB | GPU Mem | GPU%  | Disk"
        )
    else:
        header = (
            "Timestamp            | CPU Total | CPU User | CPU Sys  | "
            "Mem Used | Python RSS | Disk MB/s"
        )
    sep = "-" * len(header)

    print(header)
    print(sep)

    if output_file:
        log_lines.append(f"SeisProc Resource Monitor - Started {datetime.datetime.now()}")
        log_lines.append(f"Interval: {interval}s, GPU: {show_gpu}")
        log_lines.append("")
        log_lines.append(header)
        log_lines.append(sep)

    try:
        iteration = 0
        while True:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Get stats
            cpu_total, cpu_user, cpu_sys = get_cpu_usage()
            mem_used, mem_total = get_memory_usage()
            python_procs = get_python_processes()
            disk_mb, disk_tps = get_disk_io()

            # Calculate Python total RSS
            python_rss = sum(p['rss_gb'] for p in python_procs)
            python_cpu = sum(p['cpu'] for p in python_procs)

            # GPU stats
            gpu_mem = 0.0
            gpu_usage = 0.0
            if show_gpu:
                gpu_pytorch = get_gpu_memory_pytorch()
                gpu_apple = get_gpu_stats_apple()
                gpu_mem = gpu_pytorch.get('allocated_gb', 0.0)
                gpu_usage = gpu_apple.get('gpu_usage', 0.0)

            # Format line
            if show_gpu:
                line = (
                    f"{timestamp} | "
                    f"{cpu_total:5.1f} | "
                    f"{mem_used:6.1f} | "
                    f"{python_rss:8.2f} | "
                    f"{gpu_mem:7.2f} | "
                    f"{gpu_usage:5.1f} | "
                    f"{disk_mb:5.1f}"
                )
            else:
                line = (
                    f"{timestamp} | "
                    f"{cpu_total:8.1f}% | "
                    f"{cpu_user:7.1f}% | "
                    f"{cpu_sys:7.1f}% | "
                    f"{mem_used:7.1f}GB | "
                    f"{python_rss:9.2f}GB | "
                    f"{disk_mb:8.1f}"
                )

            print(line)

            if output_file:
                log_lines.append(line)

            # Every 10 iterations, show detailed info
            if iteration % 10 == 0:
                details = []

                if python_procs:
                    details.append(f"  Python: {len(python_procs)} proc, CPU {python_cpu:.1f}%, RSS {python_rss:.2f}GB")

                if show_gpu and gpu_pytorch.get('available'):
                    details.append(f"  GPU: {gpu_pytorch['device']}, Allocated: {gpu_mem:.2f}GB")

                    # Show GPU processes
                    gpu_procs = get_metal_gpu_processes()
                    if gpu_procs:
                        top_proc = max(gpu_procs, key=lambda x: x['cpu'])
                        details.append(f"  Top GPU proc: {top_proc['cmd']} (CPU {top_proc['cpu']:.1f}%)")

                for detail in details:
                    print(detail)
                    if output_file:
                        log_lines.append(detail)

            # Check duration
            if duration and (time.time() - start_time) >= duration:
                print(f"\nDuration limit ({duration}s) reached.")
                break

            iteration += 1
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")

    # Write to file
    if output_file:
        log_lines.append("")
        log_lines.append(f"Monitoring ended: {datetime.datetime.now()}")
        log_lines.append(f"Total duration: {time.time() - start_time:.1f}s")

        with open(output_file, 'w') as f:
            f.write('\n'.join(log_lines))
            f.write('\n')
        print(f"\nLog saved to: {output_file}")

    # Print summary
    elapsed = time.time() - start_time
    print(f"\nTotal monitoring time: {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description='Monitor system resources during SeisProc execution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/resource_monitor.py                    # Basic monitoring, 5s interval
  python tools/resource_monitor.py -i 2              # 2 second interval
  python tools/resource_monitor.py -o run.log        # Save to file
  python tools/resource_monitor.py -d 3600 -o run.log  # Run for 1 hour
  python tools/resource_monitor.py --no-gpu          # Disable GPU monitoring

For detailed Apple GPU stats (requires sudo):
  sudo powermetrics --samplers gpu_power -n 1
        """
    )
    parser.add_argument(
        '-i', '--interval',
        type=float,
        default=5,
        help='Monitoring interval in seconds (default: 5)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output log file (default: console only)'
    )
    parser.add_argument(
        '-d', '--duration',
        type=int,
        default=None,
        help='Maximum monitoring duration in seconds'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU monitoring'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Just check system capabilities and exit'
    )

    args = parser.parse_args()

    # Just check capabilities
    if args.check:
        print("System Capabilities Check")
        print("=" * 40)

        # CPU
        cpu_total, cpu_user, cpu_sys = get_cpu_usage()
        print(f"CPU: {cpu_total:.1f}% (user: {cpu_user:.1f}%, sys: {cpu_sys:.1f}%)")

        # Memory
        mem_used, mem_total = get_memory_usage()
        print(f"Memory: {mem_used:.1f}GB used / {mem_total:.1f}GB total")

        # GPU PyTorch
        gpu_pytorch = get_gpu_memory_pytorch()
        print(f"PyTorch GPU: {gpu_pytorch['device']} ({gpu_pytorch['driver']})")
        if gpu_pytorch['available']:
            print(f"  Allocated: {gpu_pytorch['allocated_gb']:.2f}GB")

        # Apple GPU
        gpu_apple = get_gpu_stats_apple()
        print(f"Apple GPU: {gpu_apple['method']}")
        if gpu_apple['gpu_usage'] > 0:
            print(f"  Usage: {gpu_apple['gpu_usage']:.1f}%")
            print(f"  Power: {gpu_apple['gpu_power']:.1f}mW")

        # Disk
        disk_mb, disk_tps = get_disk_io()
        print(f"Disk I/O: {disk_mb:.1f} MB/s, {disk_tps:.0f} TPS")

        print("=" * 40)
        return

    monitor_loop(
        interval=args.interval,
        output_file=args.output,
        duration=args.duration,
        show_gpu=not args.no_gpu
    )


if __name__ == '__main__':
    main()
