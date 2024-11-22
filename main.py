import os
import shutil
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

# Set random seed for reproducibility
random.seed(42)

annotations_dir = 'annotations/'
list_file = os.path.join(annotations_dir, 'list.txt')
mask_dir = os.path.join(annotations_dir, 'trimaps/')
img_dir = 'images/'

# Output directories
dataset_dir = 'custom_dataset/'
train_img_dir = os.path.join(dataset_dir, 'images/train/')
val_img_dir = os.path.join(dataset_dir, 'images/val/')
test_img_dir = os.path.join(dataset_dir, 'images/test/')
train_label_dir = os.path.join(dataset_dir, 'labels/train/')
val_label_dir = os.path.join(dataset_dir, 'labels/val/')
test_label_dir = os.path.join(dataset_dir, 'labels/test/')

# Ensure directories exist
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(test_img_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)

# Step 1: Parse 'list.txt' to map image names to class IDs and splits
print("Parsing 'list.txt' to map image names to class IDs and splits...")

image_info_list = []

with open(list_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue  # Skip comments and empty lines
        parts = line.strip().split()
        if len(parts) < 4:
            print(f"Skipping malformed line: {line}")
            continue  # Skip lines that don't have enough columns
        image_name = parts[0]
        class_id = int(parts[1]) - 1  # Convert class IDs to 0-based index
        species_id = int(parts[2])    # Not used here
        split = int(parts[3])         # 1=trainval, 2=test

        image_info_list.append({'image_name': image_name, 'class_id': class_id, 'split': split})

print(f"Total images parsed: {len(image_info_list)}")

# Create dictionaries for class mapping and splits
image_to_class_id = {}
trainval_images = []
test_images = []

for info in image_info_list:
    image_to_class_id[info['image_name']] = info['class_id']
    if info['split'] == 1:
        trainval_images.append(info['image_name'])
    elif info['split'] == 2:
        test_images.append(info['image_name'])
    else:
        print(f"Unknown split for image {info['image_name']}. Defaulting to trainval.")
        trainval_images.append(info['image_name'])

print(f"Total trainval images: {len(trainval_images)}")
print(f"Total test images: {len(test_images)}")

# Step 2: Generate labels from segmentation masks
print("\nGenerating labels from segmentation masks...")

# Possible extensions to check
possible_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

# Function to process each mask file
def process_mask_file(image_name, split):
    mask_file = image_name + '.png'
    # Paths to mask and image files
    mask_path = os.path.join(mask_dir, mask_file)
    if not os.path.exists(mask_path):
        print(f"Mask file for {image_name} not found. Skipping.")
        return

    # Find the correct image file
    img_path = None
    for ext in possible_extensions:
        temp_path = os.path.join(img_dir, image_name + ext)
        if os.path.exists(temp_path):
            img_path = temp_path
            break

    if img_path is None:
        print(f"Image file for {image_name} not found. Skipping.")
        return

    # Get class ID
    if image_name not in image_to_class_id:
        print(f"Image {image_name} not found in class mapping. Skipping.")
        return
    class_id = image_to_class_id[image_name]

    # Open mask and image
    try:
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        img = Image.open(img_path)
    except Exception as e:
        print(f"Error opening image or mask for {image_name}: {e}")
        return

    img_width, img_height = img.size

    # Convert mask to NumPy array
    mask_np = np.array(mask)

    # Create binary mask for foreground pixels (value == 1 or 3)
    # According to dataset: 1: Foreground, 2: Background, 3: Not classified
    foreground = (mask_np == 1) | (mask_np == 3)

    # Check if foreground is empty
    if not np.any(foreground):
        print(f"No foreground found in mask {mask_file}. Skipping.")
        return

    # Get coordinates of foreground pixels
    coords = np.column_stack(np.where(foreground))
    if coords.size == 0:
        print(f"No valid coordinates found in mask {mask_file}. Skipping.")
        return

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Compute normalized coordinates
    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    bbox_width = (x_max - x_min) / img_width
    bbox_height = (y_max - y_min) / img_height

    # Ensure coordinates are within [0, 1]
    x_center = np.clip(x_center, 0, 1)
    y_center = np.clip(y_center, 0, 1)
    bbox_width = np.clip(bbox_width, 0, 1)
    bbox_height = np.clip(bbox_height, 0, 1)

    # Create label content
    label_content = f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n"

    # Determine destination directories
    if split == 1:
        # For trainval, we'll split further into train and val
        # Let's do an 80-20 split
        if random.random() < 0.8:
            img_dest_dir = train_img_dir
            label_dest_dir = train_label_dir
        else:
            img_dest_dir = val_img_dir
            label_dest_dir = val_label_dir
    elif split == 2:
        img_dest_dir = test_img_dir
        label_dest_dir = test_label_dir
    else:
        # Default to train
        img_dest_dir = train_img_dir
        label_dest_dir = train_label_dir

    # Copy image to destination directory
    img_dest_path = os.path.join(img_dest_dir, image_name + '.jpg')
    shutil.copy(img_path, img_dest_path)

    # Write label file
    label_path = os.path.join(label_dest_dir, image_name + '.txt')
    with open(label_path, 'w') as f:
        f.write(label_content)

# Process all images
print("\nProcessing images and generating labels...")
for info in tqdm(image_info_list, desc='Processing Images'):
    process_mask_file(info['image_name'], info['split'])

print("Processing complete.")

# Step 3: Create 'dataset.yaml' file
print("\nCreating 'dataset.yaml' file...")

dataset_yaml_content = f"""
train: ../custom_dataset/images/train
val: ../custom_dataset/images/val
test: ../custom_dataset/images/test

is_coco: False

nc: 37  # number of classes

names: ['Abyssinian', 'American_Bulldog', 'American_Pit_Bull_Terrier', 'Basset_Hound', 'Beagle', 'Bengal', 'Birman',
        'Bombay', 'Boxer', 'British_Shorthair', 'Chihuahua', 'Egyptian_Mau', 'English_Cocker_Spaniel', 'English_Setter',
        'German_Shorthaired_Pointer', 'Great_Pyrenees', 'Havanese', 'Japanese_Chin', 'Keeshond', 'Leonberger',
        'Maine_Coon', 'Miniature_Pinscher', 'Newfoundland', 'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian_Blue',
        'Saint_Bernard', 'Samoyed', 'Scottish_Terrier', 'Shiba_Inu', 'Siamese', 'Sphynx', 'Staffordshire_Bull_Terrier',
        'Wheaten_Terrier', 'Yorkshire_Terrier']
"""

dataset_yaml_path = os.path.join('data', 'dataset.yaml')
os.makedirs('data', exist_ok=True)
with open(dataset_yaml_path, 'w') as f:
    f.write(dataset_yaml_content.strip())

print(f"'dataset.yaml' file created at {dataset_yaml_path}")

print("\nAll steps completed successfully.")