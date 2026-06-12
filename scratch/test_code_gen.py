from tools import CodeGeneratorTool
import os

tool = CodeGeneratorTool("TestCodeGenerator", "Generates code.")

print("Test 1: Simple python script")
res1 = tool.execute("Write a Python function to calculate the factorial of a number.", ["Backend"])
print(res1)
print("=" * 60)

print("Test 2: Saving code to a file")
res2 = tool.execute("Write a simple hello world Python script and save to scratch/hello.py", ["Backend"])
print(res2)
print("=" * 60)

if os.path.exists("scratch/hello.py"):
    print("Found hello.py, contents:")
    with open("scratch/hello.py", "r") as f:
        print(f.read())
else:
    print("hello.py was not created.")
