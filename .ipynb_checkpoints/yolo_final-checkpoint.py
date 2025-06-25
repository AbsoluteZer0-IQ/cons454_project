from ultralytics import YOLO
import os, sys
import glob
import argparse

parser = argparse.ArgumentParser(description="Classify and sort images into folders")
parser.add_argument('--input_folder', type=str, required=True, help="Path to folder with images to classify")
parser.add_argument('--output_folder', type=str, default="sorted_images", help="Where to save sorted images")
args = parser.parse_args()

model = YOLO("yolov8n.pt")
os.makedirs(args.output_folder, exist_ok=True)
image_paths = glob.glob(os.path.join(args.input_folder, "*.*"))

person_dir = os.path.join(args.output_folder, "people")
os.makedirs(person_dir, exist_ok=True)
animal_dir = os.path.join(args.output_folder, "animals")
os.makedirs(animal_dir, exist_ok=True)
empty_dir = os.path.join(args.output_folder, "nothing")
os.makedirs(empty_dir, exist_ok=True)

for image_path in image_paths:
    filename = os.path.basename(image_path)
    results = model(image_path)
    for result in results:
        class_ids = result.boxes.cls.tolist()
        labels = [model.names[int(cls_id)] for cls_id in class_ids]
        if "person" in labels:
            result_path = os.path.join(person_dir, filename)
        elif len(result.boxes) == 0:
            result_path = os.path.join(empty_dir, filename)
        else:
            result_path = os.path.join(animal_dir, filename)
    results[0].save(filename=result_path)