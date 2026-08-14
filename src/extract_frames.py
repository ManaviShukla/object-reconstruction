import cv2 as cv2
# import json
import pandas as pd

cap = cv2.VideoCapture("data/chair_001/raw/vid.mp4")

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()


sample_interval_ms = 600
next_sample_time_ms = 0
metadata = []
frame_count = 0
while True:
    ret, frame = cap.read()

    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

    if not ret:
        break   # No more frames → end of video

    if timestamp_ms >= next_sample_time_ms:
        print(frame_count, timestamp_ms)
        next_sample_time_ms += sample_interval_ms
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        cv2.imwrite(f"data/chair_001/frames/frame_{frame_count:04d}.jpg", frame)
        metadata.append({
            "frame_index": f"{frame_count:04d}",
            "timestamp": timestamp_ms,
            "sharpness": sharpness,
            "source_frame_number": cap.get(cv2.CAP_PROP_POS_FRAMES)
        })
        frame_count += 1

    cv2.imshow("Video", frame)

    # Press Q to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
df = pd.DataFrame(metadata)
df.to_csv("data/chair_001/frames/metadata.csv", index=False)
print(f"Number of frames extracted: {df.shape[0]}\n")
print(f"Minimum Sharpness: {df['sharpness'].min()}\n")
print(f"Median Sharpness: {df['sharpness'].median()}\n")
print(f"Maximum Sharpness: {df['sharpness'].max()}\n")
# Release resources
cap.release()
cv2.destroyAllWindows()