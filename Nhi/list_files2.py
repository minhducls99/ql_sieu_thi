# -*- coding: utf-8 -*-
import os
import sys

# Set stdout to utf-8
sys.stdout.reconfigure(encoding='utf-8')

files = os.listdir(r"C:\Nhi")
for f in files:
    print(f)