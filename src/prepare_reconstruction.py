from pathlib import Path
import pandas as pd
import shutil

DATA_DIR = Path("data/chair_001")
FRAME_DIR = DATA_DIR / "frames"
OUTPUT_DIR = DATA_DIR / "colmap_input"


metadata = pd.read_csv(FRAME_DIR / "metadata.csv")
# print(metadata.columns)
# print(metadata.head(5))


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.glob("*.jpg")

SHARPNESS_THRESHOLD = 15
rejected = []
for row in metadata.itertuples():
    frame_index = row.frame_index
    frame_file = FRAME_DIR / f"frame_{frame_index}.jpg"
    if row.sharpness >= SHARPNESS_THRESHOLD:
        
        if not frame_file.exists():
            print(f"Frame file {frame_file} does not exist")
            continue
        output_file = OUTPUT_DIR / f"frame_{frame_index}.jpg"
        shutil.copy2(frame_file, output_file)
    else:
        rejected.append((frame_file, row.sharpness))


print("Total sampled frames:", len(metadata))
print("Selected:", len(list(OUTPUT_DIR.glob("*.jpg"))))
print("Rejected:", len(rejected))
[print(f, s, "\n") for f, s in rejected]
