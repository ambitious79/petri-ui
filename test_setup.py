#!/usr/bin/env python3
"""
Quick test script to verify petri-ui setup.
Run this after installation to check if everything is working.
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    tests = [
        ("flask", "Flask"),
        ("flask_cors", "flask-cors"),
        ("inspect_ai", "Inspect AI"),
        ("petri", "Petri"),
        ("petri.solvers.auditor_agent", "Petri auditor_agent"),
        ("petri.scorers.judge", "Petri judge"),
    ]
    
    failed = []
    for module, name in tests:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {str(e)}")
            failed.append(name)
    
    return len(failed) == 0, failed


def test_api_keys():
    """Check if API keys are set."""
    print("\nChecking API keys...")
    
    keys = {
        "CHUTES_API_KEY": "Chutes API",
    }
    
    missing = []
    for key, name in keys.items():
        if os.environ.get(key):
            print(f"  ✓ {name} is set")
        else:
            print(f"  ⚠ {name} is not set (optional but needed for evaluations)")
            missing.append(name)
    
    return len(missing) == 0, missing


def test_directories():
    """Check if required directories exist."""
    print("\nChecking directories...")
    
    dirs = ["outputs", "logs", "temp", "static"]
    missing = []
    
    for dir_name in dirs:
        if os.path.exists(dir_name):
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ⚠ {dir_name}/ not found (will be created on first run)")
            missing.append(dir_name)
    
    return True, missing


def test_files():
    """Check if required files exist."""
    print("\nChecking files...")
    
    files = [
        "app.py",
        "petri_ui_task.py",
        "requirements.txt",
        "config.example.json",
    ]
    
    missing = []
    for file_name in files:
        if os.path.exists(file_name):
            print(f"  ✓ {file_name}")
        else:
            print(f"  ✗ {file_name} not found")
            missing.append(file_name)
    
    return len(missing) == 0, missing


def test_petri_ui_task():
    """Test if the petri_ui_task can be loaded."""
    print("\nTesting petri_ui_task...")
    
    try:
        from petri_ui_task import petri_ui_audit
        print("  ✓ petri_ui_audit task can be imported")
        return True, []
    except Exception as e:
        print(f"  ✗ Failed to import petri_ui_audit: {str(e)}")
        return False, [str(e)]


def main():
    """Run all tests."""
    print("=" * 60)
    print("Petri-UI Setup Verification")
    print("=" * 60)
    print()
    
    all_passed = True
    
    # Run tests
    tests = [
        ("Package imports", test_imports),
        ("API keys", test_api_keys),
        ("Directories", test_directories),
        ("Required files", test_files),
        ("Petri UI task", test_petri_ui_task),
    ]
    
    results = []
    for test_name, test_func in tests:
        passed, issues = test_func()
        results.append((test_name, passed, issues))
        if not passed and test_name in ["Package imports", "Required files", "Petri UI task"]:
            all_passed = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for test_name, passed, issues in results:
        if passed:
            print(f"✓ {test_name}: PASSED")
        else:
            print(f"✗ {test_name}: FAILED")
            if issues:
                for issue in issues:
                    print(f"    - {issue}")
    
    print()
    
    if all_passed:
        print("🎉 All critical tests passed!")
        print("\nYou can now start the server:")
        print("  python app.py --debug")
        print("\nOr run with PM2:")
        print("  pm2 start ecosystem.config.js")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Run: pip install -r requirements.txt")
        print("  - Run: bash setup_petri.sh")
        return 1


if __name__ == "__main__":
    sys.exit(main())


