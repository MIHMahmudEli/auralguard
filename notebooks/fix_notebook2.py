import sys

with open('kaggle_training.ipynb', 'r', encoding='utf-8') as f:
    text = f.read()

# The json.loads fails at char 4572 with "Invalid control character"
# This means a literal newline is inside a JSON string value.
# Find it: look for sequences where a \n appears inside "..." 
# without being preceded by backslash.

# Strategy: scan for JSON strings and find unescaped newlines
# A simpler approach: the error is at char 4572, let's look at context
ctx_start = max(0, 4572 - 100)
ctx_end = min(len(text), 4572 + 100)
context = text[ctx_start:ctx_end]

# Show the surrounding 200 chars
print("Context around error position 4572:")
print(repr(context))
print()

# Try to find the position by matching pattern
# The error is "Invalid control character" - likely a raw newline in a string
# Find all positions of raw newlines that might be inside JSON strings
in_string = False
escape = False
fixes = []
for i, ch in enumerate(text[:4600]):
    if escape:
        escape = False
        continue
    if ch == '\\':
        escape = True
        continue
    if ch == '"' and not escape:
        # Toggle string state (but need to handle \" properly - already handled by escape flag)
        # Actually this is too simplistic for JSON
        pass

# Simpler: just read the file as bytes and patch
print("Trying byte-level fix...")
with open('kaggle_training.ipynb', 'rb') as f:
    data = f.read()

# The error at char 4572 - find what's there
# Look for patterns of real newlines inside JSON strings
# A JSON string starts with " and ends with " (unescaped)
# Inside, \n should be \n (the JSON escape)
# But if there's a real 0x0a byte inside a string, it's invalid

# Find the issue by scanning for newlines that aren't part of \n escapes
idx = 4572
# Show bytes around that position
print(f"Bytes at error: {repr(data[idx:idx+20])}")
print(f"Bytes before: {repr(data[idx-10:idx])}")
