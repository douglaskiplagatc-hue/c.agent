# calculator_extra.py
# Additional calculator functions beyond the basic arithmetic operations.


def square(n):
    """Return the square of n (n * n).
    
    Args:
        n: A number to be squared
        
    Returns:
        The square of n
    """
    return n * n


if __name__ == "__main__":
    # Simple test block
    print("Testing square function...")
    
    # Test positive numbers
    print("square(2) =", square(2))
    print("square(3) =", square(3))
    print("square(5) =", square(5))
    
    # Test zero
    print("square(0) =", square(0))
    
    # Test negative numbers
    print("square(-1) =", square(-1))
    print("square(-3) =", square(-3))
    
    print("All tests completed.")