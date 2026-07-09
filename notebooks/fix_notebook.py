import sys, json

with open('kaggle_training.ipynb', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line with MISSING
for i, line in enumerate(lines):
    if 'MISSING' in line:
        print(f'Line {i+1}: missing comma, fixing...', file=sys.stderr)
        # This line should end with , but doesn't
        # Add comma before the line ending
        lines[i] = line.rstrip('\r\n') + ',\n'
        break

with open('kaggle_training.ipynb', 'w', encoding='utf-8') as f:
    f.writelines(lines)

# Validate
json.loads(open('kaggle_training.ipynb').read())
print('Valid JSON OK', file=sys.stderr)
