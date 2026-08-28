def add(a, b):
    return a + b
def subtract(a, b):
    return a - b
def multiply(a, b):
    return a * b
def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
def power(a, b):
    """Calculate the power of a number.
    
    Args:
        a (int or float): The base number.
        b (int or float): The exponent.
    
    Returns:
        int or float: The result of a raised to the power of b.
    """
    return a ** b