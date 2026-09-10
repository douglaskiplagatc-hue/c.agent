import os


def check_directory(directory):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    files = os.listdir(directory)
    for file in files:
        file_path = os.path.join(directory, file)

        # Check for empty directories
        if os.path.isdir(file_path) and len(os.listdir(file_path)) == 0:
            print(f"Empty directory: {file_path}")

        # File integrity check (simple attempt to read the file)
        try:
            with open(file_path, 'r'):
                pass
        except OSError as e:
            print(f"File {file_path} is corrupted: {e}")

        # Permissions check
        if not os.access(file_path, os.R_OK | os.W_OK):
            print(f"Inaccessible file: {file_path}")

        # Metadata check (mod time)
        modification_time = os.path.getmtime(file_path)
        print(f"Modification time of {file_path}: {modification_time}")
