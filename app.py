import os

def rename_files_to_numbers(folder_path):
    # Get the list of files in the folder
    files = os.listdir(folder_path)
    
    # Sort files to ensure consistent numbering
    files.sort()

    # Rename files to numbers
    for index, file_name in enumerate(files, start=1):
        old_path = os.path.join(folder_path, file_name)
        file_extension = os.path.splitext(file_name)[1]

        new_name = f"{index}{file_extension}"
        new_path = os.path.join(folder_path, new_name)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path} -> {new_path}")

# Replace 'dataset' with the actual path to your folder
folder_path = 'train/output'

# Call the function to rename files
rename_files_to_numbers(folder_path)