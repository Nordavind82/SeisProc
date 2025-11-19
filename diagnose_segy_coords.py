#!/usr/bin/env python3
"""Diagnose SEGY coordinate encoding and FK calculations."""
import segyio
import numpy as np

segy_path = "/Users/olegadamovich/Downloads/npr3_field.sgy"

print("=" * 80)
print("SEGY COORDINATE ANALYSIS")
print("=" * 80)

with segyio.open(segy_path, ignore_geometry=True) as f:
    # Read first gather (assume first 100 traces)
    n_traces = min(100, len(f.trace))

    print(f"\nFirst {n_traces} traces:")
    print(f"{'Trace':<6} {'Offset':<10} {'RcvStn':<12} {'RcvLine':<10} {'ScalCo':<8} {'SrcX':<12} {'SrcY':<12} {'RcvX':<12} {'RcvY':<12}")
    print("-" * 100)

    offsets = []
    rcv_stations = []
    rcv_lines = []
    scalcos = []
    src_x = []
    src_y = []
    rcv_x = []
    rcv_y = []

    for i in range(n_traces):
        header = f.header[i]

        offset = header[segyio.TraceField.offset]
        rcv_station = header[segyio.TraceField.ReceiverGroupElevation]  # Often misused for station

        # Try different fields for receiver info
        try:
            rcv_group = header[segyio.TraceField.GroupNumber]
        except:
            rcv_group = 0

        scalco = header[segyio.TraceField.SourceGroupScalar]
        sx = header[segyio.TraceField.SourceX]
        sy = header[segyio.TraceField.SourceY]
        gx = header[segyio.TraceField.GroupX]
        gy = header[segyio.TraceField.GroupY]

        offsets.append(offset)
        rcv_stations.append(rcv_station)
        rcv_lines.append(rcv_group)
        scalcos.append(scalco)
        src_x.append(sx)
        src_y.append(sy)
        rcv_x.append(gx)
        rcv_y.append(gy)

        if i < 10:
            print(f"{i:<6} {offset:<10} {rcv_station:<12} {rcv_group:<10} {scalco:<8} {sx:<12} {sy:<12} {gx:<12} {gy:<12}")

    offsets = np.array(offsets)
    rcv_x = np.array(rcv_x)
    rcv_y = np.array(rcv_y)
    src_x = np.array(src_x)
    src_y = np.array(src_y)
    scalco = scalcos[0] if scalcos else 0

    print("\n" + "=" * 80)
    print("OFFSET ANALYSIS")
    print("=" * 80)

    offset_diffs = np.abs(np.diff(offsets))
    offset_diffs_nonzero = offset_diffs[offset_diffs > 0]

    print(f"\nFirst 10 offsets: {offsets[:10]}")
    print(f"Offset differences: {offset_diffs[:10]}")
    print(f"Median offset step: {np.median(offset_diffs_nonzero):.1f}")

    # Check if in feet
    median_step = np.median(offset_diffs_nonzero)
    if median_step > 50:
        print(f"\n⚠️ Offsets likely in FEET (median step = {median_step:.1f})")
        print(f"   In meters: {median_step * 0.3048:.1f} m")

    print("\n" + "=" * 80)
    print("COORDINATE ANALYSIS")
    print("=" * 80)

    print(f"\nScalar (scalco): {scalco}")
    print(f"Source X range: {src_x.min()} to {src_x.max()}")
    print(f"Source Y range: {src_y.min()} to {src_y.max()}")
    print(f"Receiver X range: {rcv_x.min()} to {rcv_x.max()}")
    print(f"Receiver Y range: {rcv_y.min()} to {rcv_y.max()}")

    # Apply scalar
    if scalco < 0:
        scale_factor = abs(scalco)
        rcv_x_scaled = rcv_x / scale_factor
        src_x_scaled = src_x / scale_factor
    elif scalco > 0:
        scale_factor = scalco
        rcv_x_scaled = rcv_x * scale_factor
        src_x_scaled = src_x * scale_factor
    else:
        rcv_x_scaled = rcv_x
        src_x_scaled = src_x
        scale_factor = 1

    print(f"\nAfter applying scalar ({scalco}):")
    print(f"Receiver X (scaled): {rcv_x_scaled[:10]}")

    # Calculate receiver spacing
    rcv_spacing = np.abs(np.diff(rcv_x_scaled))
    rcv_spacing_nonzero = rcv_spacing[rcv_spacing > 0]

    if len(rcv_spacing_nonzero) > 0:
        print(f"\nReceiver X spacing:")
        print(f"  First 10 spacings: {rcv_spacing[:10]}")
        print(f"  Median: {np.median(rcv_spacing_nonzero):.2f} m")

        dx = np.median(rcv_spacing_nonzero)
    else:
        print(f"\n⚠️ Cannot calculate receiver spacing from X coordinates")
        dx = None

    print("\n" + "=" * 80)
    print("FK WAVENUMBER CALCULATION")
    print("=" * 80)

    if dx is not None:
        k_nyquist = 1 / (2 * dx)
        print(f"\nUsing dx = {dx:.2f} m:")
        print(f"  Nyquist wavenumber: {k_nyquist:.6f} cycles/m")
        print(f"  Nyquist wavenumber: {k_nyquist * 1000:.3f} mcycles/m")

        # What if we see -2.5 to 2.5 in the plot?
        observed_k_max = 2.5
        implied_dx = 1 / (2 * observed_k_max)

        print(f"\nIf FK plot shows ±{observed_k_max} cycles/m:")
        print(f"  Implied dx would be: {implied_dx:.3f} m")
        print(f"  Actual dx: {dx:.2f} m")
        print(f"  Ratio: {dx / implied_dx:.1f}x")

        if dx / implied_dx > 10:
            print(f"\n⚠️ MAJOR DISCREPANCY! FK plot is using wrong spacing!")
