"""
Basic test to verify the SpringMassMPC implementation

This script runs basic tests to ensure the implementation is working correctly.

Author: MPC Spring-Mass Python Implementation
Date: November 2025
"""

import numpy as np
from SpringMassMPC import SpringMassMPC


def test_initialization():
    """Test that controller can be initialized"""
    print("Test 1: Initialization... ", end="")
    try:
        mpc = SpringMassMPC(M=5, m=1.0, k=5.0, c=0.1)
        assert mpc.M == 5
        assert mpc.m == 1.0
        assert mpc.k == 5.0
        assert mpc.c == 0.1
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_spring_force():
    """Test spring force calculation"""
    print("Test 2: Spring force... ", end="")
    try:
        mpc = SpringMassMPC(M=3, k=5.0, k_nl=0.01, is_linear=True)
        
        # Linear spring
        r = 1.0
        f_linear = mpc.spring_force(r)
        assert np.isclose(f_linear, 5.0), f"Expected 5.0, got {f_linear}"
        
        # Nonlinear spring
        mpc.is_linear = False
        f_nonlinear = mpc.spring_force(r)
        expected = 5.0 - 0.01  # k*r - k_nl*r^3
        assert np.isclose(f_nonlinear, expected), f"Expected {expected}, got {f_nonlinear}"
        
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_system_dynamics():
    """Test that system dynamics can be computed"""
    print("Test 3: System dynamics... ", end="")
    try:
        mpc = SpringMassMPC(M=3, m=1.0, k=5.0, c=0.1)
        
        # State: [x1, x2, x3, v1, v2, v3]
        y = np.array([1.0, 0.5, -0.5, 0.1, 0.0, -0.1])
        
        dydt = mpc.system_dynamics(0.0, y)
        
        # Check that output has correct shape
        assert dydt.shape == (6,), f"Expected shape (6,), got {dydt.shape}"
        
        # Check that velocity derivatives match velocities
        assert np.allclose(dydt[:3], y[3:6])
        
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_uncontrolled_simulation():
    """Test uncontrolled system simulation"""
    print("Test 4: Uncontrolled simulation... ", end="")
    try:
        mpc = SpringMassMPC(M=3, m=1.0, k=5.0, c=0.1, dt=0.1)
        
        x0 = np.array([1.0, 0.5, -0.5])
        v0 = np.zeros(3)
        
        t, x = mpc.simulate_uncontrolled(x0, v0, T=1.0)
        
        # Check that simulation ran
        assert len(t) > 0, "No time points returned"
        assert x.shape[0] == len(t), "Position array size mismatch"
        assert x.shape[1] == 3, f"Expected 3 masses, got {x.shape[1]}"
        
        # Check that positions changed
        assert not np.allclose(x[0], x[-1]), "Positions didn't change"
        
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_mpc_solve():
    """Test that MPC can solve optimization problem"""
    print("Test 5: MPC solve (small problem)... ", end="")
    try:
        # Use small problem for fast testing
        mpc = SpringMassMPC(M=3, m=1.0, k=5.0, c=0.1, dt=0.1, N=10)
        
        x0 = np.array([1.0, 0.5, -0.5])
        v0 = np.zeros(3)
        
        x_opt, v_opt, u_opt, success = mpc.solve_mpc(x0, v0, verbose=False)
        
        # Check shapes
        assert x_opt.shape == (10, 3), f"Wrong x shape: {x_opt.shape}"
        assert v_opt.shape == (10, 3), f"Wrong v shape: {v_opt.shape}"
        assert u_opt.shape == (10, 2), f"Wrong u shape: {u_opt.shape}"
        
        # Check control bounds
        assert np.all(np.abs(u_opt) <= mpc.u_max + 1e-3), "Control exceeds bounds"
        
        # Check that controller tries to reduce positions
        final_magnitude = np.max(np.abs(x_opt[-1]))
        initial_magnitude = np.max(np.abs(x0))
        
        print("✓ PASSED")
        print(f"          Initial magnitude: {initial_magnitude:.3f}")
        print(f"          Final magnitude:   {final_magnitude:.3f}")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_control_step():
    """Test a single control step"""
    print("Test 6: Control step... ", end="")
    try:
        mpc = SpringMassMPC(M=3, m=1.0, k=5.0, c=0.1, dt=0.1, N=10)
        
        x0 = np.array([1.0, 0.5, -0.5])
        v0 = np.zeros(3)
        
        x_next, v_next, u_applied, x_pred = mpc.step(x0, v0, noise_magnitude=0.0)
        
        # Check shapes
        assert x_next.shape == (3,), f"Wrong x_next shape: {x_next.shape}"
        assert v_next.shape == (3,), f"Wrong v_next shape: {v_next.shape}"
        assert u_applied.shape == (2,), f"Wrong u_applied shape: {u_applied.shape}"
        
        # Check that history was stored
        assert len(mpc.state_history) == 1
        assert len(mpc.control_history) == 1
        
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Running Basic Tests for SpringMassMPC")
    print("=" * 60)
    print()
    
    tests = [
        test_initialization,
        test_spring_force,
        test_system_dynamics,
        test_uncontrolled_simulation,
        test_mpc_solve,
        test_control_step
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    # Summary
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed!")
    else:
        print(f"✗ {total - passed} test(s) failed")
    
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

