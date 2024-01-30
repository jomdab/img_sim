import Augmentor
import os

# Specify the path to your image folder
image_folder = "train"

# Create an Augmentor pipeline for the images in the specified folder
pipeline = Augmentor.Pipeline(image_folder)

# Define the augmentation operations you want to apply
pipeline.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
pipeline.flip_left_right(probability=0.5)
pipeline.zoom_random(probability=0.5, percentage_area=0.8)

# Set the number of augmented images you want to generate
num_augmented_images = 2500

# Execute the augmentation process
pipeline.sample(num_augmented_images)
