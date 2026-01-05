"""
Comprehensive test suite for 3D Minesweeper DQN project.
Runs all test scripts in sequence to validate system functionality.

Test Coverage:
1. Action encoding consistency across components
2. Win/loss tracking accuracy
3. Hybrid mode action selection rates
4. Safe action detection availability
5. Safe action correctness validation
6. Training loop simulation
7. Model Q-value diagnostics

Usage:
    python run_all_tests.py
    
Exit Codes:
    0: All tests passed
    1: One or more tests failed
"""

import subprocess
import sys
import time
from pathlib import Path

# ANSI color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

# Test scripts to run in order (all in tests/ subdirectory)
TEST_SCRIPTS = [
    {
        'name': 'Action Encoding Test',
        'file': 'tests/test_action_encoding.py',
        'description': 'Validates action space encoding consistency'
    },
    {
        'name': 'Win Tracking Test',
        'file': 'tests/test_win_tracking.py',
        'description': 'Tests win/loss detection accuracy (Bayesian agent)'
    },
    {
        'name': 'Safe Action Legal Test',
        'file': 'tests/test_safe_legal.py',
        'description': 'Validates safe action correctness'
    },
    {
        'name': 'Safe Action Detection Test',
        'file': 'tests/test_safe_detection.py',
        'description': 'Measures safe action availability rates'
    },
    {
        'name': 'Hybrid Mode Test',
        'file': 'tests/test_hybrid_mode.py',
        'description': 'Tests hybrid action selection behavior (DQN)'
    },
    {
        'name': 'Training Wins Simulation',
        'file': 'tests/debug_training_wins.py',
        'description': 'Simulates training loop win tracking'
    },
    {
        'name': 'DQN Diagnostics',
        'file': 'tests/diagnose_dqn.py',
        'description': 'Compares Q-values between pretrained/trained models',
        'optional': True  # Requires checkpoint files
    },
]


def run_test(test_info):
    """Run a single test script and return result."""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}Running: {test_info['name']}{RESET}")
    print(f"{BLUE}Description: {test_info['description']}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")
    
    start_time = time.time()
    
    try:
        # Get the project root directory (parent of tests/ directory)
        project_root = Path(__file__).parent.absolute()
        
        # Use venv Python if available, otherwise use current Python
        venv_python = project_root / "venv" / "Scripts" / "python.exe"
        python_executable = str(venv_python) if venv_python.exists() else sys.executable
        
        # Run the test script with PYTHONPATH set to project root
        import os
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root)
        
        result = subprocess.run(
            [python_executable, test_info['file']],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per test
            env=env,
            cwd=str(project_root)  # Run from project root
        )
        
        elapsed = time.time() - start_time
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        # Check result
        if result.returncode == 0:
            print(f"\n{GREEN}✓ {test_info['name']} PASSED{RESET} ({elapsed:.1f}s)")
            return True
        elif result.returncode == 2:
            print(f"\n{YELLOW}⊘ {test_info['name']} SKIPPED{RESET} ({elapsed:.1f}s)")
            return None  # None = skipped
        else:
            print(f"\n{RED}✗ {test_info['name']} FAILED{RESET} (exit code: {result.returncode}, {elapsed:.1f}s)")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n{RED}✗ {test_info['name']} TIMEOUT{RESET} (exceeded 5 minutes)")
        return False
    except FileNotFoundError:
        if test_info.get('optional'):
            print(f"\n{YELLOW}⊘ {test_info['name']} SKIPPED{RESET} (file not found - optional test)")
            return None  # None means skipped
        else:
            print(f"\n{RED}✗ {test_info['name']} ERROR{RESET} (file not found: {test_info['file']})")
            return False
    except Exception as e:
        print(f"\n{RED}✗ {test_info['name']} ERROR{RESET} ({str(e)})")
        return False


def main():
    """Run all tests and report results."""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}3D MINESWEEPER DQN - COMPREHENSIVE TEST SUITE{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")
    
    # Verify we're in the right directory
    required_files = ['backend/minesweeper.py', 'agents/dueling_cnn_agent.py']
    for file in required_files:
        if not Path(file).exists():
            print(f"{RED}Error: Required file not found: {file}{RESET}")
            print(f"{RED}Please run this script from the project root directory.{RESET}")
            sys.exit(1)
    
    results = []
    passed = 0
    failed = 0
    skipped = 0
    
    start_time = time.time()
    
    # Run each test
    for test in TEST_SCRIPTS:
        result = run_test(test)
        results.append((test['name'], result))
        
        if result is True:
            passed += 1
        elif result is False:
            failed += 1
        else:  # None = skipped
            skipped += 1
    
    total_time = time.time() - start_time
    
    # Print summary
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}TEST SUMMARY{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")
    
    for name, result in results:
        if result is True:
            status = f"{GREEN}✓ PASSED{RESET}"
        elif result is False:
            status = f"{RED}✗ FAILED{RESET}"
        else:
            status = f"{YELLOW}⊘ SKIPPED{RESET}"
        print(f"  {status}  {name}")
    
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"Total: {len(results)} tests | "
          f"{GREEN}{passed} passed{RESET} | "
          f"{RED}{failed} failed{RESET} | "
          f"{YELLOW}{skipped} skipped{RESET}")
    print(f"Time: {total_time:.1f}s")
    print(f"{BLUE}{'='*70}{RESET}\n")
    
    # Exit with appropriate code
    if failed > 0:
        print(f"{RED}TEST SUITE FAILED{RESET}")
        sys.exit(1)
    else:
        print(f"{GREEN}TEST SUITE PASSED{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
