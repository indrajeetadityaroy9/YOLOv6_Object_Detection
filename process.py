import os
import xml.etree.ElementTree as ET
import shutil
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm
import yaml

annotations_dir = 'Annotation/'
images_dir = 'Images/'
labels_dir = 'labels/'
os.makedirs(labels_dir, exist_ok=True)

custom_dataset_dir = 'custom_dataset'
train_images_dir = os.path.join(custom_dataset_dir, 'images', 'train')
val_images_dir = os.path.join(custom_dataset_dir, 'images', 'val')
test_images_dir = os.path.join(custom_dataset_dir, 'images', 'test')
train_labels_dir = os.path.join(custom_dataset_dir, 'labels', 'train')
val_labels_dir = os.path.join(custom_dataset_dir, 'labels', 'val')
test_labels_dir = os.path.join(custom_dataset_dir, 'labels', 'test')

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

breed_dirs = [d for d in os.listdir(annotations_dir) if os.path.isdir(os.path.join(annotations_dir, d))]

class_names = sorted([d.split('-', 1)[1] for d in breed_dirs])
print("Class Names:", class_names)
class_to_id = {breed_name: idx for idx, breed_name in enumerate(class_names)}
print("Class ids:", class_to_id)

all_image_paths = []
all_label_paths = []

for breed_dir in tqdm(breed_dirs, desc='Processing breeds'):
    breed_annotation_dir = os.path.join(annotations_dir, breed_dir)
    breed_images_dir = os.path.join(images_dir, breed_dir)
    breed_name = breed_dir.split('-', 1)[1]
    class_id = class_to_id[breed_name]

    if not os.path.exists(breed_images_dir):
        print(f"Images directory {breed_images_dir} does not exist.")
        continue

    for annotation_file in os.listdir(breed_annotation_dir):
        annotation_path = os.path.join(breed_annotation_dir, annotation_file)
        if not os.path.isfile(annotation_path):
            continue

        try:
            tree = ET.parse(annotation_path)
            root = tree.getroot()
        except ET.ParseError as e:
            print(f"Error parsing {annotation_path}: {e}")
            continue

        base_filename = os.path.splitext(annotation_file)[0]

        image_filename = f"{base_filename}.jpg"

        image_path = os.path.join(breed_images_dir, image_filename)

        if not os.path.exists(image_path):
            print(f"Image {image_path} not found.")
            continue

        label_filename = f"{base_filename}.txt"
        label_path = os.path.join(labels_dir, label_filename)

        size = root.find('size')
        if size is None:
            print(f"No size information in {annotation_path}")
            continue
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        with open(label_path, 'w') as f:
            for obj in root.findall('object'):
                bbox = obj.find('bndbox')
                if bbox is None:
                    print(f"No bounding box in {annotation_path}")
                    continue

                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)

                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                bbox_width = (xmax - xmin) / img_width
                bbox_height = (ymax - ymin) / img_height

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

        all_image_paths.append(image_path)
        all_label_paths.append(label_path)

assert len(all_image_paths) == len(all_label_paths), "Mismatch between images and labels."

train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
    all_image_paths, all_label_paths, test_size=0.30, random_state=42, shuffle=True
)

val_imgs, test_imgs, val_labels, test_labels = train_test_split(
    temp_imgs, temp_labels, test_size=0.50, random_state=42, shuffle=True
)

print(f"Total images: {len(all_image_paths)}")
print(f"Training set: {len(train_imgs)}")
print(f"Validation set: {len(val_imgs)}")
print(f"Test set: {len(test_imgs)}")


def copy_and_rename_files(image_list, label_list, split):
    if split == 'train':
        image_dest = train_images_dir
        label_dest = train_labels_dir
    elif split == 'val':
        image_dest = val_images_dir
        label_dest = val_labels_dir
    elif split == 'test':
        image_dest = test_images_dir
        label_dest = test_labels_dir
    else:
        raise ValueError("Split must be 'train', 'val', or 'test'.")

    for idx, (img_src, lbl_src) in tqdm(enumerate(zip(image_list, label_list)), total=len(image_list),
                                        desc=f'Copying {split} data'):
        new_image_name = f"{split}{idx}.jpg"
        new_label_name = f"{split}{idx}.txt"

        img_dst = os.path.join(image_dest, new_image_name)
        lbl_dst = os.path.join(label_dest, new_label_name)

        shutil.copyfile(img_src, img_dst)
        shutil.copyfile(lbl_src, lbl_dst)


copy_and_rename_files(train_imgs, train_labels, 'train')
copy_and_rename_files(val_imgs, val_labels, 'val')
copy_and_rename_files(test_imgs, test_labels, 'test')

data_yaml_path = os.path.join(custom_dataset_dir, 'data.yaml')
with open(data_yaml_path, 'w') as outfile:
    yaml_content = {
        'train': os.path.abspath(train_images_dir),
        'val': os.path.abspath(val_images_dir),
        'test': os.path.abspath(test_images_dir),
        'nc': len(class_names),
        'names': class_names,
        'is_coco': False
    }

    yaml.dump(yaml_content, outfile, default_flow_style=False)

print(f"data.yaml has been created at {data_yaml_path}")
