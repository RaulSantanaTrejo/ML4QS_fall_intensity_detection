import os

def find_and_replace_files(directory, filename, replacement):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == filename:
                filepath = os.path.join(root, file)
                replace_first_line(filepath, replacement)

def replace_first_line(filepath, replacement):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    if len(lines) > 0:
        lines[0] = replacement + "\n"

    with open(filepath, 'w') as file:
        file.writelines(lines)


if __name__ == '__main__':
    # Example usage:
    directory = os.getcwd()  # Current directory
    filename = "Pressure.csv"  # File name to search for

    replacement = '"Time (s)","Pressure (hPa)"'  # Line replacement text

    find_and_replace_files(directory, filename, replacement)
