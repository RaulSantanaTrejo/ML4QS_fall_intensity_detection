import os

def rename_csv_files(directory, filename, new_filename):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == filename:
                filepath = os.path.join(root, file)
                new_filepath = os.path.join(root, new_filename)
                rename_file(filepath, new_filepath)

def rename_file(filepath, new_filepath):
    os.rename(filepath, new_filepath)

if __name__ == '__main__':
    # Example usage:
    directory = os.getcwd()  # Current directory
    filename = "Barometer.csv"  # File name to search for
    new_filename = "Pressure.csv"  # New filename to rename to

    rename_csv_files(directory, filename, new_filename)
