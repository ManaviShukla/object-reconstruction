import re
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "data/chair_001/reconstruction/database.db"

conn = sqlite3.connect(DB_PATH)

images = pd.read_sql_query("""
SELECT image_id, name
FROM images""", conn)

pairs = pd.read_sql_query("""
SELECT pair_id, rows AS verified_matches
FROM two_view_geometries
WHERE rows > 0
ORDER BY rows DESC""", conn)

print("Images:", len(images))
print("Verified image pairs:", len(pairs))

print("\nVerified-match statistics:")
print(pairs["verified_matches"].describe())

# print("\nTop 10 pairs:")
# print(pairs.head(10))

MAX_IMAGE_ID = 2147483647

def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) // MAX_IMAGE_ID
    return int(image_id1), int(image_id2)

image_names = dict(
    zip(images["image_id"], images["name"])
)

pairs[["image_id1", "image_id2"]] = pairs["pair_id"].apply(
    lambda x: pd.Series(pair_id_to_image_ids(x))
)

pairs["image1"] = pairs["image_id1"].map(image_names)
pairs["image2"] = pairs["image_id2"].map(image_names)

print(
    pairs[
        ["image1", "image2", "verified_matches"]
    ].head(20)
)
camera = pd.read_sql_query(
    "SELECT * FROM cameras",
    conn
)

# print("\nCamera:")
# print(camera)
def frame_number(name):
    return int(re.search(r"(\d+)", name).group(1))

pairs["frame_gap"] = (
    pairs["image1"].apply(frame_number)
    - pairs["image2"].apply(frame_number)
).abs()

candidates = pairs[
    (pairs["verified_matches"] >= 100) &
    (pairs["frame_gap"] >= 2)
].sort_values(
    ["verified_matches"],
    ascending=False
)

print(
    candidates[
        [
            "image_id1",
            "image_id2",
            "image1",
            "image2",
            "frame_gap",
            "verified_matches"
        ]
    ].head(20)
)


camera_params = np.frombuffer(
    camera.iloc[0]["params"],
    dtype=np.float64
)

print("Camera params:", camera_params)
conn.close()