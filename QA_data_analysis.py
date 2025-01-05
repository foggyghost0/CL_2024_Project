import csv
import re


def analyze_scores(csv_path):
    groups = {
        ("with lora", "Black"): [],
        ("with lora", "East Asian"): [],
        ("no lora", "Black"): [],
        ("no lora", "East Asian"): [],
    }
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            face_name = row["face_name"]
            if re.search(r"/with_", face_name):
                group = "with lora"
            elif re.search(r"/without_", face_name):
                group = "no lora"
            else:
                continue
            race = row["race"]
            if race not in ["Black", "East Asian"]:
                continue
            # Convert space-separated floats into a list
            scores_str = (
                row["race_scores"].replace("\n", " ").replace("[", "").replace("]", "")
            )
            scores = [float(x) for x in scores_str.split()]
            # Index for Black is 1, East Asian is 3
            index = 1 if race == "Black" else 3
            groups[(group, race)].append(scores[index])
    for (grp, race), vals in groups.items():
        if vals:
            avg_score = sum(vals) / len(vals)
            print(f"{grp} - {race}: {avg_score:.6f}")


if __name__ == "__main__":
    analyze_scores("/Users/bohdan/Documents/CL_2024_Project/predictions_for_QA.csv")
