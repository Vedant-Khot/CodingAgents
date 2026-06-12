#!/usr/bin/env python3
"""
hello_world.py
A simple hello world script that demonstrates basic Python functionality.
"""

import os


def create_hello_script():
    """
    Creates a simple hello world Python script at scratch/hello.py
    """
    # Define the content of the hello world script
    hello_script_content = '''#!/usr/bin/env python3
"""
hello.py
A simple hello world script.
"""

def main():
    """Prints a hello world message."""
    print("Hello, World!")

if __name__ == "__main__":
    main()
'''

    # Define the directory and file path
    scratch_dir = "scratch"
    file_path = os.path.join(scratch_dir, "hello.py")

    # Create the scratch directory if it doesn't exist
    os.makedirs(scratch_dir, exist_ok=True)
    print(f"Directory '{scratch_dir}' created or already exists.")

    # Write the hello world script to the file
    with open(file_path, "w") as file:
        file.write(hello_script_content)

    print(f"Hello world script created at: {file_path}")


if __name__ == "__main__":
    create_hello_script()