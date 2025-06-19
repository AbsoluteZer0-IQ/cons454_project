import argparse
import os
import torch
from torchvision import transforms, models
from PIL import Image
import joblib
import numpy as np
import shutil

parser = argparse.ArgumentParser(description="Classify and sort images into folders")
parser.add_argument('--input_folder', type=str, required=True, help="Path to folder with images to classify")
parser.add_argument('--output_folder', type=str, default="sorted_images", help="Where to save sorted images")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_extractor = models.densenet121(weights="DenseNet121_Weights.IMAGENET1K_V1")
feature_extractor.classifier = torch.nn.Identity()
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

pipe = joblib.load("final_model.pkl")

class_names = ["nothing", "something"]

transform = transforms.Compose([
    transforms.Resize((400, 711)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

os.makedirs(args.output_folder, exist_ok=True)

for fname in os.listdir(args.input_folder):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(args.input_folder, fname)
    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = feature_extractor(input_tensor)
    
    features_np = features.cpu().numpy()
    pred = int(pipe.predict(features_np)[0])
    pred_label = class_names[pred]

    class_dir = os.path.join(args.output_folder, pred_label)
    os.makedirs(class_dir, exist_ok=True)

    shutil.copy(img_path, os.path.join(class_dir, fname))