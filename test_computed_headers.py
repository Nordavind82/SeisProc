"""
Test script for computed headers functionality.
Demonstrates how to use trace header math during SEGY import.
"""

from utils.segy_import.header_mapping import HeaderMapping, StandardHeaders
from utils.segy_import.computed_headers import ComputedHeaderField


def test_computed_headers_basic():
    """Test basic computed header functionality."""
    print("=" * 60)
    print("Test 1: Basic Computed Header Configuration")
    print("=" * 60)

    # Create header mapping with standard headers
    mapping = HeaderMapping()
    mapping.add_standard_headers(StandardHeaders.get_minimal())

    # Add computed headers
    # Example 1: Convert receiver station to receiver line
    receiver_line = ComputedHeaderField(
        name='receiver_line',
        expression='round(receiver_x / 1000)',
        description='Receiver line number computed from X coordinate',
        format='i'
    )
    mapping.add_computed_field(receiver_line)

    # Example 2: Compute absolute offset
    abs_offset = ComputedHeaderField(
        name='abs_offset',
        expression='abs(offset)',
        description='Absolute value of offset',
        format='i'
    )
    mapping.add_computed_field(abs_offset)

    # Example 3: Compute scaled coordinate
    scaled_x = ComputedHeaderField(
        name='scaled_source_x',
        expression='source_x * 0.01',
        description='Scaled source X coordinate',
        format='f'
    )
    mapping.add_computed_field(scaled_x)

    print(f"\nHeader Mapping: {mapping}")
    print(f"  Raw headers: {len(mapping.fields)}")
    print(f"  Computed headers: {len(mapping.computed_fields)}")

    # Test computation with sample data
    sample_headers = {
        'trace_sequence_file': 1,
        'cdp': 1000,
        'offset': -250,
        'receiver_x': 150000,
        'source_x': 100000,
        'sample_count': 2000,
        'sample_interval': 2000
    }

    print("\nSample raw headers:")
    for key, value in sample_headers.items():
        print(f"  {key}: {value}")

    # Get processor and compute
    processor = mapping.get_computed_processor()
    computed = processor.compute_headers(sample_headers, trace_idx=0)

    print("\nComputed headers:")
    for key, value in computed.items():
        print(f"  {key}: {value}")

    print("\n✓ Test 1 passed!")


def test_computed_headers_chaining():
    """Test chained computed headers (computed headers referencing other computed headers)."""
    print("\n" + "=" * 60)
    print("Test 2: Chained Computed Headers")
    print("=" * 60)

    mapping = HeaderMapping()
    mapping.add_standard_headers(StandardHeaders.get_minimal())

    # Create chained computations
    # Step 1: Compute scaled coordinate
    scaled_x = ComputedHeaderField(
        name='scaled_x',
        expression='source_x / 1000',
        description='Scaled X in kilometers',
        format='f'
    )
    mapping.add_computed_field(scaled_x)

    # Step 2: Use scaled coordinate in another computation
    rounded_x = ComputedHeaderField(
        name='rounded_x',
        expression='round(scaled_x)',
        description='Rounded scaled X',
        format='i'
    )
    mapping.add_computed_field(rounded_x)

    # Step 3: Use both in a final computation
    result = ComputedHeaderField(
        name='result',
        expression='rounded_x * 10 + 5',
        description='Final computed result',
        format='i'
    )
    mapping.add_computed_field(result)

    print(f"\nChained computations:")
    for field in mapping.computed_fields:
        deps = field.get_dependencies()
        print(f"  {field.name} = {field.expression}")
        if deps:
            print(f"    Dependencies: {deps}")

    # Test execution order
    processor = mapping.get_computed_processor()
    print(f"\nExecution order: {[f.name for f in processor.execution_order]}")

    # Compute with sample data
    sample_headers = {
        'source_x': 12345,
        'cdp': 1000,
        'offset': 100,
        'sample_count': 2000,
        'sample_interval': 2000
    }

    computed = processor.compute_headers(sample_headers, trace_idx=0)

    print(f"\nInput: source_x = {sample_headers['source_x']}")
    print(f"Computed:")
    print(f"  scaled_x = {computed['scaled_x']:.3f}")
    print(f"  rounded_x = {computed['rounded_x']}")
    print(f"  result = {computed['result']}")

    print("\n✓ Test 2 passed!")


def test_computed_headers_error_handling():
    """Test error handling in computed headers."""
    print("\n" + "=" * 60)
    print("Test 3: Error Handling")
    print("=" * 60)

    mapping = HeaderMapping()
    mapping.add_standard_headers(StandardHeaders.get_minimal())

    # Add computed header with potential errors
    risky_computation = ComputedHeaderField(
        name='risky',
        expression='cdp / offset',  # Will fail if offset is 0
        description='CDP divided by offset (may have division by zero)',
        format='f'
    )
    mapping.add_computed_field(risky_computation)

    processor = mapping.get_computed_processor()

    # Test with valid data
    valid_headers = {'cdp': 1000, 'offset': 100}
    result = processor.compute_headers(valid_headers, trace_idx=0)
    print(f"\nValid computation: cdp={valid_headers['cdp']}, offset={valid_headers['offset']}")
    print(f"  Result: {result['risky']}")

    # Test with zero offset (should return 0 and log error)
    invalid_headers = {'cdp': 1000, 'offset': 0}
    result = processor.compute_headers(invalid_headers, trace_idx=1)
    print(f"\nInvalid computation: cdp={invalid_headers['cdp']}, offset={invalid_headers['offset']}")
    print(f"  Result (should be 0): {result['risky']}")

    # Get error summary
    print(f"\n{processor.get_error_summary()}")

    print("\n✓ Test 3 passed!")


def test_computed_headers_math_functions():
    """Test various math functions."""
    print("\n" + "=" * 60)
    print("Test 4: Math Functions")
    print("=" * 60)

    mapping = HeaderMapping()

    # Test various math functions
    functions = [
        ('floor_test', 'floor(offset / 100)', 'Floor division'),
        ('ceil_test', 'ceil(offset / 100)', 'Ceiling division'),
        ('sqrt_test', 'sqrt(abs(offset))', 'Square root of absolute offset'),
        ('min_test', 'min(cdp, 1000)', 'Minimum of cdp and 1000'),
        ('max_test', 'max(cdp, 1000)', 'Maximum of cdp and 1000'),
        ('power_test', 'offset ** 2', 'Offset squared'),
        ('modulo_test', 'cdp % 100', 'CDP modulo 100'),
    ]

    for name, expr, desc in functions:
        mapping.add_computed_field(ComputedHeaderField(name, expr, desc, 'f'))

    processor = mapping.get_computed_processor()

    # Test with sample data
    sample_headers = {'cdp': 1234, 'offset': 350}

    computed = processor.compute_headers(sample_headers, trace_idx=0)

    print(f"\nInput: cdp={sample_headers['cdp']}, offset={sample_headers['offset']}")
    print(f"\nComputed results:")
    for name, _, desc in functions:
        print(f"  {name:15s} = {computed[name]:12.2f}  # {desc}")

    print("\n✓ Test 4 passed!")


def test_save_load_mapping():
    """Test saving and loading mapping with computed headers."""
    print("\n" + "=" * 60)
    print("Test 5: Save/Load Mapping")
    print("=" * 60)

    # Create mapping with computed headers
    mapping = HeaderMapping()
    mapping.add_standard_headers(StandardHeaders.get_minimal())
    mapping.set_ensemble_keys(['cdp'])

    mapping.add_computed_field(ComputedHeaderField(
        'receiver_line',
        'round(receiver_x / 1000)',
        'Receiver line from X coordinate',
        'i'
    ))

    mapping.add_computed_field(ComputedHeaderField(
        'source_line',
        'round(source_x / 1000)',
        'Source line from X coordinate',
        'i'
    ))

    # Save to file
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name

    try:
        mapping.save_to_file(temp_file)
        print(f"\nSaved mapping to: {temp_file}")
        print(f"  Raw headers: {len(mapping.fields)}")
        print(f"  Computed headers: {len(mapping.computed_fields)}")
        print(f"  Ensemble keys: {mapping.ensemble_keys}")

        # Load back
        loaded_mapping = HeaderMapping.load_from_file(temp_file)
        print(f"\nLoaded mapping from: {temp_file}")
        print(f"  Raw headers: {len(loaded_mapping.fields)}")
        print(f"  Computed headers: {len(loaded_mapping.computed_fields)}")
        print(f"  Ensemble keys: {loaded_mapping.ensemble_keys}")

        # Verify computed headers
        print(f"\nComputed headers loaded:")
        for field in loaded_mapping.computed_fields:
            print(f"  {field.name}: {field.expression}")

        # Test that it works
        processor = loaded_mapping.get_computed_processor()
        sample_headers = {'receiver_x': 150000, 'source_x': 100000, 'cdp': 1000}
        computed = processor.compute_headers(sample_headers, trace_idx=0)
        print(f"\nTest computation:")
        print(f"  Input: receiver_x={sample_headers['receiver_x']}, source_x={sample_headers['source_x']}")
        print(f"  Output: receiver_line={computed['receiver_line']}, source_line={computed['source_line']}")

        print("\n✓ Test 5 passed!")

    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Testing Computed Headers Functionality")
    print("=" * 60)

    try:
        test_computed_headers_basic()
        test_computed_headers_chaining()
        test_computed_headers_error_handling()
        test_computed_headers_math_functions()
        test_save_load_mapping()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
