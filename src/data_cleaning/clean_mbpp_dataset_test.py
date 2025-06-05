from .clean_mbpp_dataset import check_valid_indentation, check_valid_colon

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

# Test cases for colon validation
colon_test_cases = [
    # Valid cases - no space before colon
    ("""def hello():
    print("Hello")""", True, "def with proper colon"),
    
    ("""if True:
    print("Hello")""", True, "if with proper colon"),
    
    ("""for i in range(10):
    print(i)""", True, "for with proper colon"),
    
    ("""while True:
    break""", True, "while with proper colon"),
    
    ("""try:
    pass
except:
    pass""", True, "try/except with proper colon"),
    
    ("""class MyClass:
    pass""", True, "class with proper colon"),
    
    ("""else:
    print("else")""", True, "else with proper colon"),
    
    ("""elif condition:
    pass""", True, "elif with proper colon"),
    
    ("""finally:
    cleanup()""", True, "finally with proper colon"),
    
    ("""with open('file') as f:
    pass""", True, "with with proper colon"),
    
    # Invalid cases - space before colon
    ("""def hello() :
    print("Hello")""", False, "def with space before colon"),
    
    ("""if True :
    print("Hello")""", False, "if with space before colon"),
    
    ("""for i in range(10) :
    print(i)""", False, "for with space before colon"),
    
    ("""while True :
    break""", False, "while with space before colon"),
    
    ("""try :
    pass
except :
    pass""", False, "try/except with space before colon"),
    
    ("""class MyClass :
    pass""", False, "class with space before colon"),
    
    ("""else :
    print("else")""", False, "else with space before colon"),
    
    ("""elif condition :
    pass""", False, "elif with space before colon"),
    
    ("""finally :
    cleanup()""", False, "finally with space before colon"),
    
    ("""with open('file') as f :
    pass""", False, "with with space before colon"),
    
    # Mixed valid and invalid
    ("""def hello():
    if True :
        print("Hello")""", False, "mixed: valid def, invalid if"),
    
    ("""def hello() :
    if True:
        print("Hello")""", False, "mixed: invalid def, valid if"),
    
    # Edge cases
    ("""# This is a comment
def hello():
    pass""", True, "code with comments"),
    
    ("""def hello():
    # Comment with colon:
    pass""", True, "comment containing colon"),
    
    ("""def hello():
    x = {'key': 'value'}
    pass""", True, "dictionary with colon"),
    
    ("""def hello():
    lambda x: x + 1
    pass""", True, "lambda with colon"),
    
    ("""print("Hello")
print("World")""", True, "no control structures"),
    
    ("", True, "empty string"),
    
    ("""
    
    """, True, "only whitespace"),
]

print("\n\nTesting colon validation:")
print("=" * 50)

for code, expected, description in colon_test_cases:
    result = check_valid_colon(code)
    status = "✓" if result == expected else "✗"
    print(f"{status} {description}: {result} (expected: {expected})")
