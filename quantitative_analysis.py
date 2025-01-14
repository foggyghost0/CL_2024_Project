from __future__ import print_function, division
import os
import pandas as pd
import warnings
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms
from PIL import Image
import dlib
import spaces
import cv2

warnings.filterwarnings("ignore")
# MENTION USINF FAIRFACE SCRIPT AS BASIS FOR THIS 

def ensure_dir(directory):
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def crop_face(image, bbox):
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]

@spaces.GPU
def predict_race_gender_age(
    save_prediction_at,
    imgs_path="prepared_faces/",
    model_path="fair_face_models/fairface_alldata_20191111.pt",
    device="mps",
    max_face_size=224
):
    ensure_dir(os.path.dirname(save_prediction_at))

    detector = dlib.get_frontal_face_detector()

    img_names = [
        os.path.join(imgs_path, x)
        for x in os.listdir(imgs_path)
        if x.endswith((".jpg", ".png"))
    ]

    model = torchvision.models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 18)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    trans = transforms.Compose(
        [
            transforms.Resize((max_face_size, max_face_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    face_names = []
    race_preds = []
    gender_preds = []
    age_preds = []
    race_scores = []

    for index, img_name in enumerate(img_names):
        if index % 100 == 0:
            print("Processing image {}/{}".format(index, len(img_names)))

        try:
            image = dlib.load_rgb_image(img_name)
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            faces = detector(gray_image, 1)

            if len(faces) == 0:
                print(f"No faces detected in {img_name}, skipping...")
                continue

            for face in faces:
                bbox = rect_to_bb(face)
                cropped_face = crop_face(image, bbox)
                pil_image = Image.fromarray(cropped_face)

                pil_image = pil_image.resize((max_face_size, max_face_size))
                image_tensor = trans(pil_image).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(image_tensor)

                outputs = outputs.cpu().numpy().squeeze()

                race_outputs = outputs[:7]
                gender_outputs = outputs[7:9]
                age_outputs = outputs[9:18]

                race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
                gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
                age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

                race_pred = np.argmax(race_score)
                gender_pred = np.argmax(gender_score)
                age_pred = np.argmax(age_score)

                face_names.append(img_name)
                race_preds.append(race_pred)
                gender_preds.append(gender_pred)
                age_preds.append(age_pred)
                race_scores.append(race_score)

        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    result = pd.DataFrame(
        {
            "face_name": face_names,
            "race_pred": race_preds,
            "gender_pred": gender_preds,
            "age_pred": age_preds,
            "race_scores": race_scores,
        }
    )

    race_labels = [
        "White",
        "Black",
        "Latino_Hispanic",
        "East Asian",
        "Southeast Asian",
        "Indian",
        "Middle Eastern",
    ]
    gender_labels = ["Male", "Female"]
    age_labels = [
        "0-2",
        "3-9",
        "10-19",
        "20-29",
        "30-39",
        "40-49",
        "50-59",
        "60-69",
        "70+",
    ]

    result["race"] = result["race_pred"].apply(lambda x: race_labels[x])
    result["gender"] = result["gender_pred"].apply(lambda x: gender_labels[x])
    result["age"] = result["age_pred"].apply(lambda x: age_labels[x])

    result.to_csv(save_prediction_at, index=False)
    print(f"Predictions saved at {save_prediction_at}")


if __name__ == "__main__":
    IMAGE_DIR = "/Users/bohdan/Documents/CL_2024_Project/images_for_quantitative_analysis"  
    OUTPUT_CSV = "/Users/bohdan/Documents/CL_2024_Project/predictions_for_QA.csv"  
    MODEL_PATH = "/Users/bohdan/Documents/CL_2024_Project/res34_fair_align_multi_7_20190809.pt"  

    DEVICE = "mps"

    ensure_dir(os.path.dirname(OUTPUT_CSV))

    predict_race_gender_age(
        save_prediction_at=OUTPUT_CSV,
        imgs_path=IMAGE_DIR,
        model_path=MODEL_PATH,
        device=DEVICE,
    )
