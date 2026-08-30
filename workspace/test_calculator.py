import sys
sys.path.insert(0, 'workspace')

from calculator import add, subtract, multiply, divide, power

tests = [
    ("add(2,3)", add(2, 3), 5),
    ("subtract(10,4)", subtract(10, 4), 6),
    ("multiply(3,4)", multiply(3, 4), 12),
    ("divide(10,2)", divide(10, 2), 5.0),
    ("power(2,3)", power(2, 3), 8),
]

all_passed = True
for expr, result, expected in tests:
    status = "PASS" if result == expected else "FAIL"
    if result != expected:
        all_passed = False
    print(f"{status}: {expr} = {result} (expected {expected})")

if all_passed:
    print("\nAll tests passed!")
    sys.exit(0)
else:
    print("\nSome tests failed!")
    sys.exit(1)