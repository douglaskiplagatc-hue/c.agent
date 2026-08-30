import sys
sys.path.insert(0, 'workspace')

from calculator_extra import square

tests = [
    ("square(2)", square(2), 4),
    ("square(3)", square(3), 9),
    ("square(0)", square(0), 0),
    ("square(-1)", square(-1), 1),
    ("square(5)", square(5), 25),
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