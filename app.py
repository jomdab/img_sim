import os

# Specify the directory path
directory = './dataset/'

# Iterate over the files in the directory
for no, filename in enumerate(os.listdir(directory), 1):
    # Check if the file is a regular file
    if os.path.isfile(os.path.join(directory, filename)):
        # Specify the new file name format
        new_filename = str(no)
        
        # Rename the file
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))