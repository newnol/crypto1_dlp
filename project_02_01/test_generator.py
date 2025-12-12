#!/usr/bin/env python3
"""
Test generator and validator for Project 02_01: Primitive Root Modulo p
Generates random test cases and validates C++ implementation against Python using gmpy2 (GMP).
"""

import os
import subprocess
import tempfile
import random
import gmpy2
from gmpy2 import mpz, is_prime, next_prime, powmod


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


def factorize(n: int, timeout_iterations: int = 100000) -> dict:
    """
    Factorize n using trial division and Pollard's rho algorithm.
    Returns a dict of {prime: exponent}.
    """
    n = mpz(n)
    factors = {}
    
    # Trial division for small factors
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes:
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
    
    # Continue with larger trial division
    p = 101
    while p * p <= n and p < 100000:
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
        p += 2
    
    # Use Pollard's rho for remaining factors
    iterations = 0
    while n > 1 and not is_prime(n) and iterations < timeout_iterations:
        factor = pollard_rho(n, max_iterations=10000)
        if factor is None:
            break
        while n % factor == 0:
            factors[int(factor)] = factors.get(int(factor), 0) + 1
            n //= factor
        iterations += 1
    
    if n > 1:
        factors[int(n)] = factors.get(int(n), 0) + 1
    
    return factors


def pollard_rho(n: int, max_iterations: int = 10000) -> int:
    """Pollard's rho algorithm for factorization."""
    n = mpz(n)
    if n % 2 == 0:
        return 2
    
    x = mpz(random.randint(2, int(n) - 1))
    y = x
    c = mpz(random.randint(1, int(n) - 1))
    d = mpz(1)
    
    iterations = 0
    while d == 1 and iterations < max_iterations:
        x = (x * x + c) % n
        y = (y * y + c) % n
        y = (y * y + c) % n
        d = gmpy2.gcd(abs(x - y), n)
        iterations += 1
    
    if d != n and d != 1:
        return int(d)
    return None


def get_prime_factors(n: int) -> list:
    """Get list of prime factors of n."""
    factors = factorize(n)
    return sorted(factors.keys())


def generate_prime_with_known_factors(bits: int, use_large_factors: bool = True) -> tuple:
    """
    Generate a prime p where p-1 has known factorization.
    Returns (p, prime_factors_of_p_minus_1).
    
    Strategy: Build p-1 = 2 * q1 * q2 * ... where qi are primes,
    then check if p = p-1 + 1 is prime.
    
    For realistic tests, we include large prime factors similar to actual test cases.
    """
    target_bits = bits
    max_attempts = 2000
    
    for _ in range(max_attempts):
        # Start with p-1 = 2
        p_minus_1 = mpz(2)
        factors = {2: 1}
        
        if use_large_factors and bits >= 64:
            # For larger primes, use a mix of small and large factors
            # This creates more realistic test cases like the actual tests
            
            # Add a few small prime factors
            small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
            num_small = random.randint(1, min(4, bits // 32))
            
            for _ in range(num_small):
                prime = random.choice(small_primes)
                if prime not in factors:
                    p_minus_1 *= prime
                    factors[prime] = 1
            
            # Add medium-sized prime factors (16-64 bits)
            current_bits = p_minus_1.bit_length()
            remaining = target_bits - current_bits
            
            if remaining > 128:
                # Add a medium prime factor
                medium_bits = random.randint(16, min(64, remaining // 3))
                medium_prime = int(next_prime(random.randint(1 << (medium_bits - 1), (1 << medium_bits) - 1)))
                p_minus_1 *= medium_prime
                factors[medium_prime] = 1
            
            # Fill the rest with a large prime factor
            current_bits = p_minus_1.bit_length()
            remaining = target_bits - current_bits
            
            if remaining > 10:
                # Generate a large prime to fill most of the remaining bits
                large_bits = remaining - random.randint(0, 5)
                if large_bits > 1:
                    low = mpz(1) << (large_bits - 1)
                    high = (mpz(1) << large_bits) - 1
                    large_prime = int(next_prime(random.randint(int(low), int(high))))
                    p_minus_1 *= large_prime
                    factors[large_prime] = 1
        else:
            # For smaller primes, use original method with small factors
            small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
            
            while p_minus_1.bit_length() < target_bits - 10:
                prime = random.choice(small_primes)
                p_minus_1 *= prime
                factors[prime] = factors.get(prime, 0) + 1
            
            # Try to reach exact bit size
            remaining_bits = target_bits - p_minus_1.bit_length()
            if remaining_bits > 1:
                low = mpz(1) << (remaining_bits - 1)
                high = (mpz(1) << remaining_bits) - 1
                q = int(next_prime(random.randint(int(low), int(high))))
                p_minus_1 *= q
                factors[q] = factors.get(q, 0) + 1
        
        p = p_minus_1 + 1
        
        if is_prime(p) and p.bit_length() >= target_bits - 5:
            return int(p), sorted(factors.keys())
    
    # Fallback: generate random prime and factorize
    p = generate_random_prime(bits)
    return p, get_prime_factors(p - 1)


def generate_random_prime(bits: int) -> int:
    """Generate a random prime with approximately the given number of bits."""
    # Generate random number with exactly 'bits' bits
    low = mpz(1) << (bits - 1)
    high = (mpz(1) << bits) - 1
    
    candidate = mpz(random.randint(int(low), int(high)))
    # Make it odd
    if candidate % 2 == 0:
        candidate += 1
    
    # Find next prime
    return int(next_prime(candidate))


def check_primitive_root(g: int, p: int, prime_factors: list) -> bool:
    """
    Check if g is a primitive root modulo p.
    g is a primitive root iff g^((p-1)/k) != 1 (mod p) for all prime factors k of (p-1).
    """
    g = mpz(g)
    p = mpz(p)
    p_minus_1 = p - 1
    
    for k in prime_factors:
        exponent = p_minus_1 // k
        if powmod(g, exponent, p) == 1:
            return False
    return True


def find_primitive_root(p: int, prime_factors: list) -> int:
    """Find a primitive root modulo p."""
    p = mpz(p)
    
    # Try small numbers first
    for g in range(2, min(int(p), 10000)):
        if check_primitive_root(g, p, prime_factors):
            return g
    
    # Random search
    for _ in range(1000):
        g = random.randint(2, int(p) - 1)
        if check_primitive_root(g, p, prime_factors):
            return g
    
    return 2  # Fallback


def find_non_primitive_root(p: int, prime_factors: list) -> int:
    """Find a non-primitive root modulo p."""
    p = mpz(p)
    
    # g = 1 is never a primitive root (unless p = 2)
    if p > 2 and not check_primitive_root(1, p, prime_factors):
        return 1
    
    # g = p - 1 has order 2
    if p > 3 and not check_primitive_root(int(p) - 1, p, prime_factors):
        return int(p) - 1
    
    # Find a primitive root and raise it to a power that gives non-primitive element
    pr = find_primitive_root(p, prime_factors)
    p_minus_1 = int(p) - 1
    
    for factor in prime_factors:
        k = p_minus_1 // factor
        g = int(powmod(mpz(pr), mpz(k), p))
        if g != 1 and not check_primitive_root(g, p, prime_factors):
            return g
    
    # Fallback: try random non-primitive roots
    for _ in range(100):
        g = random.randint(2, int(p) - 1)
        if not check_primitive_root(g, p, prime_factors):
            return g
    
    return 1


def generate_test_case(bits: int, is_primitive: bool = None) -> tuple:
    """
    Generate a test case with a prime of given bit size.
    Returns: (p, prime_factors_of_p_minus_1, g, expected_result)
    """
    # Use the fast method that generates primes with known factorization
    p, prime_factors = generate_prime_with_known_factors(bits)
    
    if is_primitive is None:
        is_primitive = random.choice([True, False])
    
    if is_primitive:
        g = find_primitive_root(p, prime_factors)
        expected = True
    else:
        g = find_non_primitive_root(p, prime_factors)
        expected = False
    
    # Double-check the expected result
    actual = check_primitive_root(g, p, prime_factors)
    
    return p, prime_factors, g, actual


def create_test_input(p: int, prime_factors: list, g: int) -> str:
    """Create test input string in the required format."""
    lines = []
    
    # Line 1: p in LSB hex
    lines.append(int_to_lsb_hex(p))
    
    # Line 2: n (number of prime factors) in LSB hex
    n = len(prime_factors)
    lines.append(int_to_lsb_hex(n))
    
    # Line 3: prime factors in LSB hex, space-separated
    factors_hex = [int_to_lsb_hex(f) for f in prime_factors]
    lines.append(' '.join(factors_hex))
    
    # Line 4: g in LSB hex
    lines.append(int_to_lsb_hex(g))
    
    return '\n'.join(lines)


def run_cpp_program(input_content: str, executable: str = "./main") -> str:
    """Run the C++ program with given input and return output."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.inp', delete=False) as inp_file:
        inp_file.write(input_content)
        inp_path = inp_file.name
    
    with tempfile.NamedTemporaryFile(mode='r', suffix='.out', delete=False) as out_file:
        out_path = out_file.name
    
    try:
        result = subprocess.run(
            [executable, inp_path, out_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"C++ program error: {result.stderr}")
            return None
        
        with open(out_path, 'r') as f:
            return f.read().strip()
    
    except subprocess.TimeoutExpired:
        print("C++ program timed out")
        return None
    
    finally:
        os.unlink(inp_path)
        os.unlink(out_path)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test generator for primitive root checker')
    parser.add_argument('--num-tests', type=int, default=100, help='Number of tests to generate')
    parser.add_argument('--executable', type=str, default='./main', help='Path to C++ executable')
    parser.add_argument('--save-tests', type=str, default=None, help='Directory to save generated tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--max-bits', type=int, default=512, help='Maximum bit size for primes')
    
    args = parser.parse_args()
    
    # Check if executable exists
    if not os.path.exists(args.executable):
        print(f"Executable not found: {args.executable}")
        print("Please compile main.cpp first: g++ -O3 -std=c++17 -o main main.cpp")
        return 1
    
    # Create save directory if specified
    if args.save_tests:
        os.makedirs(args.save_tests, exist_ok=True)
    
    # Generate tests with different bit sizes based on task requirements
    bit_sizes = []
    max_bits = args.max_bits
    
    # 40% tests with p <= 64 bits
    bit_sizes.extend([random.randint(8, 64) for _ in range(args.num_tests * 40 // 100)])
    # 30% tests with p in (64, 128] bits
    bit_sizes.extend([random.randint(65, min(128, max_bits)) for _ in range(args.num_tests * 30 // 100)])
    # 20% tests with p in (128, 256] bits
    if max_bits > 128:
        bit_sizes.extend([random.randint(129, min(256, max_bits)) for _ in range(args.num_tests * 20 // 100)])
    else:
        bit_sizes.extend([random.randint(65, max_bits) for _ in range(args.num_tests * 20 // 100)])
    # 10% tests with p in (256, 512] bits
    if max_bits > 256:
        bit_sizes.extend([random.randint(257, min(512, max_bits)) for _ in range(args.num_tests * 10 // 100)])
    else:
        bit_sizes.extend([random.randint(max(65, max_bits - 50), max_bits) for _ in range(args.num_tests * 10 // 100)])
    
    # Ensure we have exactly num_tests
    while len(bit_sizes) < args.num_tests:
        bit_sizes.append(random.randint(8, 64))
    bit_sizes = bit_sizes[:args.num_tests]
    
    random.shuffle(bit_sizes)
    
    passed = 0
    failed = 0
    errors = 0
    
    print(f"Running {args.num_tests} tests...")
    print("-" * 60)
    
    for i, bits in enumerate(bit_sizes):
        # Alternate between primitive and non-primitive roots
        is_primitive = (i % 2 == 0)
        
        try:
            p, prime_factors, g, expected = generate_test_case(bits, is_primitive)
            
            input_content = create_test_input(p, prime_factors, g)
            
            # Save test if requested
            if args.save_tests:
                test_path = os.path.join(args.save_tests, f"test_{i:03d}.inp")
                with open(test_path, 'w') as f:
                    f.write(input_content)
                
                ans_path = os.path.join(args.save_tests, f"test_{i:03d}.ans")
                with open(ans_path, 'w') as f:
                    f.write(f"{'1' if expected else '0'}\n")
            
            # Run C++ program
            cpp_output = run_cpp_program(input_content, args.executable)
            
            if cpp_output is None:
                errors += 1
                print(f"Test {i:3d}: ERROR (bits={bits})")
                continue
            
            cpp_result = cpp_output.strip() == "1"
            
            if cpp_result == expected:
                passed += 1
                if args.verbose:
                    print(f"Test {i:3d}: PASS (bits={bits}, expected={'1' if expected else '0'})")
            else:
                failed += 1
                print(f"Test {i:3d}: FAIL (bits={bits})")
                print(f"  p = {p}")
                print(f"  g = {g}")
                print(f"  factors = {prime_factors}")
                print(f"  Expected: {'1' if expected else '0'}, Got: {cpp_output}")
                
                if args.verbose:
                    print(f"  Input:\n{input_content}")
                    print()
        
        except Exception as e:
            errors += 1
            print(f"Test {i:3d}: EXCEPTION - {e}")
            import traceback
            traceback.print_exc()
    
    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed, {errors} errors out of {args.num_tests} tests")
    
    if failed == 0 and errors == 0:
        print("All tests passed!")
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
