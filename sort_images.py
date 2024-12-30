import os
import pandas as pd
import shutil

# Paths
csv_path = "/Users/bohdan/Documents/CL_2024_Project/predictions.csv"
image_dir = "/Users/bohdan/Documents/CL_2024_Project/images1024x1024/"
output_dir = "/Users/bohdan/Documents/CL_2024_Project/sorted_faces/"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load CSV data
data = pd.read_csv(csv_path)

# Filter data based on criteria
filtered_data = data[
    (data["race"].isin(["Black", "Southeast Asian"]))
    & (data["age"].isin(["20-29", "30-39", "40-49"]))
]

# Process images
for idx, row in filtered_data.iterrows():
    source_path = row["face_name"]
    if os.path.exists(source_path):
        race = row["race"].replace(" ", "_")
        destination_name = f"{race}_{idx}.png"
        destination_path = os.path.join(output_dir, destination_name)
        shutil.copy(source_path, destination_path)

print(f"Images copied and renamed to '{output_dir}'.")
