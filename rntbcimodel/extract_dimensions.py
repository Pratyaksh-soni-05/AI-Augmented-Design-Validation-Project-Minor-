import cv2
import easyocr
import pandas as pd
import numpy as np
import re
import os

# ================================
# CONFIG
# ================================
OUTPUT_FOLDER = "../demo_output"
GPU = False

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ================================
# SYMBOL NORMALIZATION
# ================================
def normalize_symbols(text):
    """
    Fix common OCR mistakes in engineering drawings
    """

    text = text.strip()

    replacements = {
        "O": "Ø",
        "0": "Ø",
        "o": "Ø",
        "Φ": "Ø",
        "φ": "Ø",
        "⌀": "Ø",
        "R ": "R",
        " r": "R",
        "r": "R"
    }

    if len(text) > 1:
        first = text[0]
        if first in replacements:
            text = replacements[first] + text[1:]

    return text


# ================================
# PREPROCESS IMAGE
# ================================
def preprocess_image(path):

    img = cv2.imread(path)

    if img is None:
        raise FileNotFoundError("Image not found")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # upscale image (improves OCR)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # remove noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # adaptive threshold works best for drawings
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10
    )

    return thresh


# ================================
# DIMENSION DETECTION REGEX
# ================================
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


# ================================
# PARSE DIMENSION
# ================================
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


# ================================
# BOUNDING BOX INFO
# ================================
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


# ================================
# MAIN EXTRACTION FUNCTION
# ================================
def extract_dimensions(image_path):

    print(f"\nProcessing image: {image_path}")

    processed_img = preprocess_image(image_path)

    reader = easyocr.Reader(['en'], gpu=GPU)

    results = reader.readtext(processed_img)

    extracted = []

    for bbox, text, conf in results:

        # Remove low confidence detections
        if conf < 0.5:
            continue

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

    return extracted


# ================================
# STANDALONE RUN
# ================================
if __name__ == "__main__":

    IMAGE_PATH = "../Dataset/drawing_old.jpg"

    dims = extract_dimensions(IMAGE_PATH)

    if len(dims) == 0:
        print("No dimensions found.")
        exit()

    df = pd.DataFrame(dims)

    print("\nExtracted Dimensions:\n")
    print(df)

    csv_path = os.path.join(OUTPUT_FOLDER, "dimensions.csv")
    json_path = os.path.join(OUTPUT_FOLDER, "dimensions.json")

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print("\nSaved files:")
    print(csv_path)
    print(json_path)