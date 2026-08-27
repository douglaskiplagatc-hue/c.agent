def greet(name):
    return "Hello " + name

if __name__ == "__main__":
    assert greet("World") == "Hello World"
    print("Test passed!")

print(greet("Douglas"))
