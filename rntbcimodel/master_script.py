import cv2
import easyocr
import pandas as pd
import numpy as np
import re
import os
import json

from symbol_utils import normalize_symbols
from mechanical_symbols import MECHANICAL_SYMBOLS

# =================================
# CONFIG
# =================================

OLD_IMAGE = "../Dataset/drawing_old.jpg"
NEW_IMAGE = "../Dataset/drawing_new.jpg"

OUTPUT_FOLDER = "../comparison_output"

GPU = False
DIST_THRESHOLD = 40
VALUE_THRESHOLD = 0.01

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

reader = easyocr.Reader(['en'], gpu=GPU)

# =================================
# DIMENSION REGEX
# =================================

DIM_PATTERN = re.compile(
    r"""
    ^
    (Ø|R|C|M)?      # optional engineering symbol
    \s*
    \d+(\.\d+)?     # number
    (\s*±\s*\d+(\.\d+)?)?   # tolerance
    (\s*(mm|cm|in))?
    $
    """,
    re.VERBOSE | re.IGNORECASE
)


def is_dimension(text):
    return bool(DIM_PATTERN.match(text.strip()))


# =================================
# PARSE DIMENSION
# =================================

def parse_dimension(text):

    symbol = None
    tolerance = None
    unit = "mm"

    t = text.strip()

    if len(t) > 0 and t[0] in ["Ø", "R", "C", "M"]:
        symbol = t[0]
        t = t[1:]

    if "±" in t:
        parts = t.split("±")
        value = float(parts[0])
        tolerance = float(parts[1])
    else:
        value = float(re.sub(r"[^\d.]", "", t))

    return symbol, value, tolerance, unit


# =================================
# IMAGE PREPROCESS
# =================================

def preprocess_image(path):

    img = cv2.imread(path)

    if img is None:
        raise FileNotFoundError(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10
    )

    return thresh


# =================================
# BOUNDING BOX INFO
# =================================

def get_bbox_info(bbox):

    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]

    x_center = sum(xs) / 4
    y_center = sum(ys) / 4

    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    return (
        round(x_center, 2),
        round(y_center, 2),
        round(width, 2),
        round(height, 2),
        int(min(xs)),
        int(min(ys)),
        int(max(xs)),
        int(max(ys))
    )


# =================================
# DIMENSION EXTRACTION
# =================================

def extract_dimensions(image_path):

    print(f"\nProcessing: {image_path}")

    processed_img = preprocess_image(image_path)

    results = reader.readtext(processed_img)

    extracted = []

    for bbox, text, conf in results:

        if conf < 0.5:
            continue

        # NORMALIZE SYMBOLS
        text = normalize_symbols(text)

        if not is_dimension(text):
            continue

        try:

            symbol, value, tol, unit = parse_dimension(text)

            (
                x_center,
                y_center,
                width,
                height,
                x_min,
                y_min,
                x_max,
                y_max
            ) = get_bbox_info(bbox)

            extracted.append({

                "raw_text": text,
                "symbol": symbol,
                "value": value,
                "tolerance": tol,
                "unit": unit,
                "confidence": round(conf, 3),

                "x_center": x_center,
                "y_center": y_center,

                "bbox_width": width,
                "bbox_height": height,

                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max
            })

        except:
            continue

    return pd.DataFrame(extracted)


# =================================
# MATCH DIMENSIONS BY LOCATION
# =================================

def find_matching_dimension(old_row, new_df):

    for idx, new_row in new_df.iterrows():

        dx = abs(old_row["x_center"] - new_row["x_center"])
        dy = abs(old_row["y_center"] - new_row["y_center"])

        if dx < DIST_THRESHOLD and dy < DIST_THRESHOLD:
            return new_row

    return None


# =================================
# DRAW RED CHANGE BOX
# =================================

def mark_changes_on_image(image_path, changes):

    img = cv2.imread(image_path)

    for row in changes:

        x1 = int(row["x_min"])
        y1 = int(row["y_min"])
        x2 = int(row["x_max"])
        y2 = int(row["y_max"])

        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),3)

        cv2.putText(
            img,
            "UPDATED",
            (x1,y1-8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,0,255),
            2
        )

    out = os.path.join(OUTPUT_FOLDER,"new_drawing_marked.png")
    cv2.imwrite(out,img)


# =================================
# TABLE DISPLAY
# =================================

def print_dimension_table(title, df):

    print("\n" + "="*70)
    print(title)
    print("="*70)

    if df.empty:
        print("No dimensions detected.")
        return

    # Ensure column order
    cols = [
        "raw_text",
        "symbol",
        "value",
        "tolerance",
        "unit",
        "confidence",
        "x_center",
        "y_center",
        "bbox_width",
        "bbox_height",
        "x_min",
        "y_min",
        "x_max",
        "y_max"
    ]

    available_cols = [c for c in cols if c in df.columns]

    print(df[available_cols].to_string(index=True))

# =================================
# MAIN PIPELINE
# =================================

def main():

    print("\nExtracting OLD drawing dimensions...")
    old_df = extract_dimensions(OLD_IMAGE)

    print_dimension_table(
        "OLD DRAWING DIMENSIONS",
        old_df
    )

    print("\nExtracting NEW drawing dimensions...")
    new_df = extract_dimensions(NEW_IMAGE)

    print_dimension_table(
        "NEW DRAWING DIMENSIONS",
        new_df
    )

    # Save raw outputs
    old_df.to_csv(os.path.join(OUTPUT_FOLDER,"old_dimensions.csv"),index=False)
    old_df.to_json(os.path.join(OUTPUT_FOLDER,"old_dimensions.json"),orient="records",indent=2)

    new_df.to_csv(os.path.join(OUTPUT_FOLDER,"new_dimensions.csv"),index=False)
    new_df.to_json(os.path.join(OUTPUT_FOLDER,"new_dimensions.json"),orient="records",indent=2)

    print("\nComparing drawings...")

    changes = []
    final_rows = []

    for _, new_row in new_df.iterrows():

        match = find_matching_dimension(new_row, old_df)

        if match is None:

            status = "added"
            changes.append(new_row.to_dict())

        else:

            if abs(new_row["value"] - match["value"]) > VALUE_THRESHOLD:

                status = "updated"
                changes.append(new_row.to_dict())

            else:
                status = "unchanged"

        row = new_row.to_dict()
        row["status"] = status

        final_rows.append(row)

    final_df = pd.DataFrame(final_rows)
    changes_df = pd.DataFrame(changes)

    print_dimension_table(
        "UPDATED / CHANGED DIMENSIONS",
        changes_df
    )

    # Save outputs
    final_df.to_csv(os.path.join(OUTPUT_FOLDER,"final_dimensions.csv"),index=False)
    final_df.to_json(os.path.join(OUTPUT_FOLDER,"final_dimensions.json"),orient="records",indent=2)

    changes_df.to_csv(os.path.join(OUTPUT_FOLDER,"changes.csv"),index=False)
    changes_df.to_json(os.path.join(OUTPUT_FOLDER,"changes.json"),orient="records",indent=2)

    mark_changes_on_image(NEW_IMAGE,changes)

    print("\nPipeline Finished.")

# =================================
# RUN SCRIPT
# =================================

if __name__ == "__main__":
    main()