import os
import cv2

# Input directory containing full camera images and output folder for crops.
# Resolve paths from this script so the command works from any current directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "samples", "digit_files_2")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "unlabeled_crops")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.isdir(INPUT_DIR):
  raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")

crop_counter = 0

for filename in os.listdir(INPUT_DIR):
  if not filename.endswith((".png", ".jpg", ".jpeg")):
    continue

  img_path = os.path.join(INPUT_DIR, filename)
  frame = cv2.imread(img_path)
  if frame is None:
    continue

  # 1. Convert to grayscale and apply inverse Otsu thresholding (Black text -> White foreground)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  _, thresh = cv2.threshold(
      gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
  )

  # 2. Find contours of potential text components
  contours, _ = cv2.findContours(
      thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )

  for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # Keep a wider range because digits 3 and 4 can be narrower or shorter.
    aspect_ratio = float(w) / h
    if 8 <= h <= 160 and 0.10 <= aspect_ratio <= 1.50:
      # Add enough context to avoid clipping thin or fragmented digits.
      pad = 8
      y1, y2 = max(0, y - pad), min(frame.shape[0], y + h + pad)
      x1, x2 = max(0, x - pad), min(frame.shape[1], x + w + pad)

      crop = gray[y1:y2, x1:x2]
      save_path = os.path.join(OUTPUT_DIR, f"crop_{crop_counter:04d}.png")
      cv2.imwrite(save_path, crop)
      crop_counter += 1

print(
    f"Extracted {crop_counter} candidate crops! Sort them into dataset_raw/1,"
    " dataset_raw/2, etc."
)