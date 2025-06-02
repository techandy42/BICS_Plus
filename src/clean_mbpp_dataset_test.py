from .clean_mbpp_dataset import check_valid_indentation

# Test cases
test_cases = [
    # Valid: 4 spaces
    ("""def hello():
    print("Hello")
    if True:
        print("World")""", True, "4 spaces"),
    
    # Valid: tabs
    ("""def hello():
\tprint("Hello")
\tif True:
\t\tprint("World")""", True, "tabs"),
    
    # Invalid: 2 spaces
    ("""def hello():
  print("Hello")
  if True:
    print("World")""", False, "2 spaces"),
    
    # Invalid: 3 spaces
    ("""def hello():
   print("Hello")""", False, "3 spaces"),
    
    # Valid: mixed tabs and spaces
    ("""def hello():
    print("Hello")
\tif True:
        print("World")""", True, "mixed tabs and spaces (valid)"),
    
    # Invalid: mixed tabs and spaces
    ("""def hello():
    print("Hello")
\t  if True:
        print("World")""", False, "mixed tabs and spaces (invalid)"),

    # Valid: no indentation
    ("""print("Hello")
print("World")""", True, "no indentation"),
    
    # Valid: empty lines don't matter
    ("""def hello():
    print("Hello")

    if True:
        print("World")""", True, "empty lines"),
]

print("Testing indentation validation:")
print("=" * 50)

for code, expected, description in test_cases:
    result = check_valid_indentation(code)
    status = "✓" if result == expected else "✗"
    print(f"{status} {description}: {result} (expected: {expected})")
