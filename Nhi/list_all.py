# -*- coding: utf-8 -*-
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Walk through all directories
for root, dirs, files in os.walk(r"C:\Nhi"):
    level = root.replace(r"C:\Nhi", '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        print(f'{subindent}{file}')