import os

# List files in C:\Nhi
files = os.listdir(r"C:\Nhi")
print("Files in C:\\Nhi:")
for f in files:
    print(f"  {f}")