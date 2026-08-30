# calculator.py
# A simple calculator module with basic arithmetic operations.


def add(a, b):
    """Return the sum of a and b."""
    return a + b


def subtract(a, b):
    """Return the difference of a and b."""
    return a - b


def multiply(a, b):
    """Return the product of a and b."""
    return a * b


def divide(a, b):
    """Return the quotient of a divided by b.
    
    Raises:
        ZeroDivisionError: If b is zero.
    """
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return a / b


def power(a, b):
    """Return a raised to the power of b."""
    return a ** b


if __name__ == "__main__":
    # Simple test block
    print("Testing calculator functions...")
    
    # Test add
    print("add(2, 3) =", add(2, 3))
    
    # Test subtract
    print("subtract(5, 2) =", subtract(5, 2))
    
    # Test multiply
    print("multiply(3, 4) =", multiply(3, 4))
    
    # Test divide
    print("divide(10, 2) =", divide(10, 2))
    
    # Test division by zero
    try:
        print("divide(10, 0) =", divide(10, 0))
    except ZeroDivisionError as e:
        print("divide(10, 0) raised ZeroDivisionError:", e)
    
    print("All tests completed.")