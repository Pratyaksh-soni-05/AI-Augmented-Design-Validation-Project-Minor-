import cv2
import pandas as pd
import numpy as np
import os

from extract_dimensions import extract_dimensions   # your previous script

# ================================
# CONFIG
# ================================

OLD_IMAGE = "../Dataset/drawing_old.jpg"
NEW_IMAGE = "../Dataset/drawing_new.jpg"

OUTPUT = "../comparison_output"
os.makedirs(OUTPUT, exist_ok=True)


# ================================
# MATCH DIMENSIONS BY LOCATION
# ================================

def find_matching_dimension(old_row, new_df):

    threshold = 40   # pixel distance

    for idx, new_row in new_df.iterrows():

        dx = abs(old_row["x_center"] - new_row["x_center"])
        dy = abs(old_row["y_center"] - new_row["y_center"])

        if dx < threshold and dy < threshold:
            return idx

    return None


# ================================
# COMPARE TABLES
# ================================

def compare_dimensions(old_df, new_df):

    updated_rows = []
    changes = []

    for i, old_row in old_df.iterrows():

        match_idx = find_matching_dimension(old_row, new_df)

        if match_idx is None:
            continue

        new_row = new_df.loc[match_idx]

        if old_row["value"] != new_row["value"]:

            changes.append(new_row)

            updated_rows.append(new_row)

        else:
            updated_rows.append(old_row)

    updated_df = pd.DataFrame(updated_rows)

    return updated_df, changes


# ================================
# DRAW RED BOX FOR CHANGES
# ================================

def mark_changes_on_image(image_path, changes):

    img = cv2.imread(image_path)

    for row in changes:

        x1 = int(row["x_min"])
        y1 = int(row["y_min"])
        x2 = int(row["x_max"])
        y2 = int(row["y_max"])

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0,0,255),
            3
        )

        cv2.putText(
            img,
            "UPDATED",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,0,255),
            2
        )

    return img


# ================================
# MAIN
# ================================

def main():

    print("Extracting OLD drawing dimensions...")
    old_dims = extract_dimensions(OLD_IMAGE)
    old_df = pd.DataFrame(old_dims)

    print("Extracting NEW drawing dimensions...")
    new_dims = extract_dimensions(NEW_IMAGE)
    new_df = pd.DataFrame(new_dims)

    old_df.to_csv(os.path.join(OUTPUT,"old_dimensions.csv"),index=False)
    new_df.to_csv(os.path.join(OUTPUT,"new_dimensions.csv"),index=False)

    print("Comparing dimensions...")

    final_df, changes = compare_dimensions(old_df,new_df)

    final_df.to_csv(
        os.path.join(OUTPUT,"final_updated_dimensions.csv"),
        index=False
    )

    print("Number of changes detected:",len(changes))

    marked = mark_changes_on_image(NEW_IMAGE,changes)

    cv2.imwrite(
        os.path.join(OUTPUT,"marked_changes.png"),
        marked
    )

    print("Finished!")


if __name__ == "__main__":
    main()