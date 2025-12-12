#!/usr/bin/env python3
"""
Test generator and validator for Project 02_02: Diffie-Hellman Key Exchange
Generates random test cases and validates C++ implementation against Python using gmpy2 (GMP).
"""

import os
import subprocess
import tempfile
import random
import gmpy2
from gmpy2 import mpz, is_prime, next_prime, powmod
from pathlib import Path


def int_to_lsb_hex(n: int) -> str:
    """
    Convert integer to LSB-to-MSB hex string format.
    Example: 255 (0xFF) -> "FF", 256 (0x100) -> "001"
    The format is: h_0*16^0 + h_1*16^1 + ... + h_k*16^k
    So we output hex digits from least significant to most significant (left to right).
    """
    if n == 0:
        return "0"
    
    result = []
    n = int(n)
    while n > 0:
        digit = n & 0xF
        if digit < 10:
            result.append(chr(ord('0') + digit))
        else:
            result.append(chr(ord('A') + digit - 10))
        n >>= 4
    
    return ''.join(result)


def lsb_hex_to_int(hex_str: str) -> int:
    """
    Convert LSB-to-MSB hex string to integer.
    Example: "FF" -> 255, "001" -> 256
    """
    result = 0
    for i, c in enumerate(hex_str):
        if '0' <= c <= '9':
            digit = ord(c) - ord('0')
        elif 'A' <= c <= 'F':
            digit = ord(c) - ord('A') + 10
        elif 'a' <= c <= 'f':
            digit = ord(c) - ord('a') + 10
        else:
            continue
        result += digit * (16 ** i)
    return result


def generate_prime(bit_size: int) -> int:
    """Generate a random prime of approximately bit_size bits."""
    # Generate a random number with the specified bit size
    low = 1 << (bit_size - 1)
    high = (1 << bit_size) - 1
    candidate = random.randint(low, high)
    # Find the next prime
    return int(next_prime(candidate))


def generate_test_case(bit_size: int) -> tuple:
    """
    Generate a Diffie-Hellman test case.
    
    Returns: (p, g, a, b, A, B, K)
    Where:
        p = prime modulus
        g = generator (random value 2 <= g < p)
        a = Alice's secret (random value 1 <= a < p)
        b = Bob's secret (random value 1 <= b < p)
        A = g^a mod p
        B = g^b mod p
        K = g^(ab) mod p = A^b mod p = B^a mod p
    """
    # Generate prime p
    p = generate_prime(bit_size)
    p_mpz = mpz(p)
    
    # Generate g (2 <= g < p)
    g = random.randint(2, p - 1)
    g_mpz = mpz(g)
    
    # Generate secrets a and b (1 <= a, b < p)
    a = random.randint(1, p - 1)
    b = random.randint(1, p - 1)
    a_mpz = mpz(a)
    b_mpz = mpz(b)
    
    # Calculate public values and shared key using GMP for accuracy
    A = int(powmod(g_mpz, a_mpz, p_mpz))
    B = int(powmod(g_mpz, b_mpz, p_mpz))
    K = int(powmod(mpz(A), b_mpz, p_mpz))
    
    return (p, g, a, b, A, B, K)


def create_test_input(p: int, g: int, a: int, b: int) -> str:
    """Create input file content in LSB hex format."""
    lines = [
        int_to_lsb_hex(p),
        int_to_lsb_hex(g),
        int_to_lsb_hex(a),
        int_to_lsb_hex(b),
    ]
    return '\n'.join(lines) + '\n'


def create_test_output(A: int, B: int, K: int) -> str:
    """Create expected output file content in LSB hex format."""
    lines = [
        int_to_lsb_hex(A),
        int_to_lsb_hex(B),
        int_to_lsb_hex(K),
    ]
    return '\n'.join(lines) + '\n'


def run_cpp_program(executable: str, input_file: str, output_file: str) -> bool:
    """Run the C++ executable with input and output files."""
    try:
        result = subprocess.run(
            [executable, input_file, output_file],
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: C++ program took too long")
        return False
    except Exception as e:
        print(f"  ERROR running C++ program: {e}")
        return False


def compare_outputs(expected_file: str, actual_file: str) -> tuple:
    """
    Compare expected and actual output files.
    Returns (match: bool, details: str)
    """
    try:
        with open(expected_file, 'r') as f:
            expected_lines = [line.strip() for line in f.readlines() if line.strip()]
        with open(actual_file, 'r') as f:
            actual_lines = [line.strip() for line in f.readlines() if line.strip()]
        
        if len(expected_lines) != len(actual_lines):
            return False, f"Line count mismatch: expected {len(expected_lines)}, got {len(actual_lines)}"
        
        for i, (exp, act) in enumerate(zip(expected_lines, actual_lines)):
            # Compare case-insensitively
            if exp.upper() != act.upper():
                exp_int = lsb_hex_to_int(exp)
                act_int = lsb_hex_to_int(act)
                return False, f"Line {i+1} mismatch:\n  Expected: {exp} ({exp_int})\n  Actual:   {act} ({act_int})"
        
        return True, "OK"
    except Exception as e:
        return False, f"Error comparing files: {e}"


def run_tests(executable: str, num_tests: int = 100, output_dir: str = None, verbose: bool = True):
    """
    Generate and run tests against the C++ executable.
    
    Distribution based on TASK.md:
    - 40 tests: p <= 64 bits
    - 30 tests: 64 < p <= 128 bits
    - 20 tests: 128 < p <= 256 bits
    - 10 tests: 256 < p <= 512 bits
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Test distribution by bit size
    test_configs = []
    
    # 40% tests with p <= 64 bits (distribute across 8, 16, 32, 48, 64 bits)
    for _ in range(8):
        test_configs.append(random.randint(8, 16))
    for _ in range(8):
        test_configs.append(random.randint(17, 32))
    for _ in range(12):
        test_configs.append(random.randint(33, 48))
    for _ in range(12):
        test_configs.append(random.randint(49, 64))
    
    # 30% tests with 64 < p <= 128 bits
    for _ in range(15):
        test_configs.append(random.randint(65, 96))
    for _ in range(15):
        test_configs.append(random.randint(97, 128))
    
    # 20% tests with 128 < p <= 256 bits
    for _ in range(10):
        test_configs.append(random.randint(129, 192))
    for _ in range(10):
        test_configs.append(random.randint(193, 256))
    
    # 10% tests with 256 < p <= 512 bits
    for _ in range(5):
        test_configs.append(random.randint(257, 384))
    for _ in range(5):
        test_configs.append(random.randint(385, 512))
    
    # Shuffle to randomize order
    random.shuffle(test_configs)
    
    # Ensure we have exactly num_tests
    test_configs = test_configs[:num_tests]
    while len(test_configs) < num_tests:
        test_configs.append(random.randint(8, 512))
    
    passed = 0
    failed = 0
    
    print(f"Running {num_tests} tests against: {executable}")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, bit_size in enumerate(test_configs):
            test_num = f"{i:03d}"
            
            # Generate test case
            p, g, a, b, A, B, K = generate_test_case(bit_size)
            
            # Create input and expected output
            input_content = create_test_input(p, g, a, b)
            expected_content = create_test_output(A, B, K)
            
            # Write files
            input_file = os.path.join(tmp_dir, f"test_{test_num}.inp")
            expected_file = os.path.join(tmp_dir, f"test_{test_num}.ans")
            actual_file = os.path.join(tmp_dir, f"test_{test_num}.out")
            
            with open(input_file, 'w') as f:
                f.write(input_content)
            with open(expected_file, 'w') as f:
                f.write(expected_content)
            
            # Save to output directory if specified
            if output_dir:
                with open(os.path.join(output_dir, f"test_{test_num}.inp"), 'w') as f:
                    f.write(input_content)
                with open(os.path.join(output_dir, f"test_{test_num}.ans"), 'w') as f:
                    f.write(expected_content)
            
            # Run C++ program
            success = run_cpp_program(executable, input_file, actual_file)
            
            if not success:
                failed += 1
                if verbose:
                    print(f"Test {test_num} ({bit_size:3d} bits): FAILED (execution error)")
                continue
            
            # Compare outputs
            match, details = compare_outputs(expected_file, actual_file)
            
            if match:
                passed += 1
                if verbose:
                    print(f"Test {test_num} ({bit_size:3d} bits): PASSED")
            else:
                failed += 1
                if verbose:
                    print(f"Test {test_num} ({bit_size:3d} bits): FAILED")
                    print(f"  Input: p={p}, g={g}, a={a}, b={b}")
                    print(f"  Expected: A={A}, B={B}, K={K}")
                    print(f"  {details}")
    
    print("=" * 60)
    print(f"Results: {passed}/{num_tests} passed, {failed}/{num_tests} failed")
    print(f"Pass rate: {100*passed/num_tests:.1f}%")
    
    return passed, failed


def validate_existing_tests(executable: str, test_dir: str, verbose: bool = True):
    """Validate the C++ executable against existing test files."""
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.inp')])
    
    passed = 0
    failed = 0
    skipped = 0
    
    print(f"Validating against existing tests in: {test_dir}")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        for inp_file in test_files:
            test_name = inp_file[:-4]  # Remove .inp extension
            out_file = test_name + '.out'
            
            input_path = os.path.join(test_dir, inp_file)
            expected_path = os.path.join(test_dir, out_file)
            actual_path = os.path.join(tmp_dir, out_file)
            
            # Check if expected output exists
            if not os.path.exists(expected_path):
                skipped += 1
                if verbose:
                    print(f"{test_name}: SKIPPED (no expected output)")
                continue
            
            # Run C++ program
            success = run_cpp_program(executable, input_path, actual_path)
            
            if not success:
                failed += 1
                if verbose:
                    print(f"{test_name}: FAILED (execution error)")
                continue
            
            # Compare outputs
            match, details = compare_outputs(expected_path, actual_path)
            
            if match:
                passed += 1
                if verbose:
                    print(f"{test_name}: PASSED")
            else:
                failed += 1
                if verbose:
                    print(f"{test_name}: FAILED - {details}")
    
    print("=" * 60)
    total_with_output = passed + failed
    print(f"Results: {passed}/{total_with_output} passed, {failed}/{total_with_output} failed, {skipped} skipped")
    if total_with_output > 0:
        print(f"Pass rate: {100*passed/total_with_output:.1f}%")
    
    return passed, failed, skipped


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test generator for Diffie-Hellman Key Exchange')
    parser.add_argument('--executable', '-e', default='./main',
                        help='Path to C++ executable (default: ./main)')
    parser.add_argument('--num-tests', '-n', type=int, default=100,
                        help='Number of tests to generate (default: 100)')
    parser.add_argument('--output-dir', '-o', default=None,
                        help='Directory to save generated tests (optional)')
    parser.add_argument('--validate', '-v', default=None,
                        help='Validate against existing tests in directory')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Reduce output verbosity')
    parser.add_argument('--seed', '-s', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
    
    # Resolve executable path
    executable = args.executable
    if not os.path.isabs(executable):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        executable = os.path.join(script_dir, executable)
    
    if not os.path.exists(executable):
        print(f"Error: Executable not found: {executable}")
        print("Please compile the C++ program first.")
        return 1
    
    verbose = not args.quiet
    
    if args.validate:
        # Validate against existing tests
        validate_existing_tests(executable, args.validate, verbose)
    else:
        # Generate and run new tests
        run_tests(executable, args.num_tests, args.output_dir, verbose)
    
    return 0


if __name__ == '__main__':
    exit(main())
