from greet import greet

result = greet("Douglas")
expected = "Welcome, Douglas"
print(f"Result: {result}")
print(f"Expected: {expected}")
print(f"Test passed: {result == expected}")

if result != expected:
    print("TEST FAILED - Need to fix greet.py")
else:
    print("TEST PASSED")
