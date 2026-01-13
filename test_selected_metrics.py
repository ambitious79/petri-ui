#!/usr/bin/env python3
"""
Test script for verifying the selected metrics functionality.

This script tests:
1. Getting current selected metrics
2. Updating selected metrics
3. Verifying the filtering logic

Usage:
    python3 test_selected_metrics.py
"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_get_metrics():
    """Test getting current selected metrics."""
    print("=" * 80)
    print("TEST 1: Get Current Selected Metrics")
    print("=" * 80)
    
    response = requests.get(f"{BASE_URL}/api/metrics")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Successfully retrieved metrics")
        print(f"  Count: {len(data['metrics'])}")
        print(f"  Metrics: {', '.join(data['metrics'])}")
        return True
    else:
        print(f"✗ Failed to get metrics: {response.status_code}")
        print(f"  Response: {response.text}")
        return False


def test_update_metrics():
    """Test updating selected metrics."""
    print("\n" + "=" * 80)
    print("TEST 2: Update Selected Metrics")
    print("=" * 80)
    
    test_metrics = [
        "unprompted_whistleblowing",
        "concerning",
        "facilitating_terrorism"
    ]
    
    payload = {
        "metrics": test_metrics
    }
    
    response = requests.post(
        f"{BASE_URL}/api/metrics",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Successfully updated metrics")
        print(f"  Count: {len(data['metrics'])}")
        print(f"  Metrics: {', '.join(data['metrics'])}")
        
        # Verify the update
        get_response = requests.get(f"{BASE_URL}/api/metrics")
        if get_response.status_code == 200:
            current_metrics = get_response.json()['metrics']
            if current_metrics == test_metrics:
                print(f"✓ Verification: Metrics successfully updated")
                return True
            else:
                print(f"✗ Verification failed: Metrics don't match")
                return False
        else:
            print(f"✗ Failed to verify update")
            return False
    else:
        print(f"✗ Failed to update metrics: {response.status_code}")
        print(f"  Response: {response.text}")
        return False


def test_restore_defaults():
    """Restore default metrics."""
    print("\n" + "=" * 80)
    print("TEST 3: Restore Default Metrics")
    print("=" * 80)
    
    default_metrics = [
        'unprompted_whistleblowing',
        'concerning',
        'disappointing',
        'cooperation_with_human_misuse',
        'unprompted_deception_toward_user',
    ]
    
    payload = {
        "metrics": default_metrics
    }
    
    response = requests.post(
        f"{BASE_URL}/api/metrics",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        print(f"✓ Successfully restored default metrics")
        return True
    else:
        print(f"✗ Failed to restore defaults: {response.status_code}")
        return False


def test_invalid_updates():
    """Test error handling for invalid updates."""
    print("\n" + "=" * 80)
    print("TEST 4: Error Handling")
    print("=" * 80)
    
    # Test empty metrics
    print("\n  Testing empty metrics array...")
    response = requests.post(
        f"{BASE_URL}/api/metrics",
        json={"metrics": []},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 400:
        print(f"  ✓ Correctly rejected empty metrics array")
    else:
        print(f"  ✗ Should have rejected empty array (got {response.status_code})")
    
    # Test missing metrics field
    print("\n  Testing missing metrics field...")
    response = requests.post(
        f"{BASE_URL}/api/metrics",
        json={},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 400:
        print(f"  ✓ Correctly rejected missing metrics field")
    else:
        print(f"  ✗ Should have rejected missing field (got {response.status_code})")
    
    # Test non-array metrics
    print("\n  Testing non-array metrics...")
    response = requests.post(
        f"{BASE_URL}/api/metrics",
        json={"metrics": "not_an_array"},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 400:
        print(f"  ✓ Correctly rejected non-array metrics")
        return True
    else:
        print(f"  ✗ Should have rejected non-array (got {response.status_code})")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("SELECTED METRICS API TEST SUITE")
    print("=" * 80)
    print(f"Testing server at: {BASE_URL}")
    print()
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/api/config", timeout=5)
        if response.status_code != 200:
            print(f"✗ Server is not responding correctly")
            print(f"  Make sure the server is running: pm2 status petri-ui")
            return 1
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to server at {BASE_URL}")
        print(f"  Error: {e}")
        print(f"  Make sure the server is running: pm2 start start-pm2.sh")
        return 1
    
    print(f"✓ Server is running\n")
    
    # Run tests
    results = []
    
    results.append(("Get Metrics", test_get_metrics()))
    results.append(("Update Metrics", test_update_metrics()))
    results.append(("Restore Defaults", test_restore_defaults()))
    results.append(("Error Handling", test_invalid_updates()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print()
    print(f"  Total: {passed}/{total} tests passed")
    print("=" * 80)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    exit(main())

