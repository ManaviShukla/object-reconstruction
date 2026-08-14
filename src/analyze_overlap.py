import pandas as pd
import cv2
import numpy as np

FRAME_DIR = "data/chair_001/frames"


# if img1 is None or img2 is None:
#     raise ValueError("Could not load frames")

def match_frames(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)



    lowe_ratio = 0.75

    lowe_filtered_matches = []

    if des1 is None or des2 is None:
        return kp1, kp2, []
    knnmatches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2)

    for match_pair in knnmatches:
        if len(match_pair)<2:
          continue
        m, n = match_pair 
        if m.distance < lowe_ratio * n.distance:
            lowe_filtered_matches.append(m)


    if des1 is None or des2 is None:
        return kp1, kp2, []

   
    return kp1, kp2, lowe_filtered_matches

def geometric_filter(kp1, kp2, matches):
    # convert matches
    pts1 = np.float32([
    kp1[m.queryIdx].pt
    for m in matches
    ])

    pts2 = np.float32([
        kp2[m.trainIdx].pt
        for m in matches
    ])

    # findFundamentalMat
    F, mask = cv2.findFundamentalMat(
    pts1,
    pts2,
    cv2.FM_RANSAC,
    1.0,
    0.99
    )
    if F is None or mask is None:
            return None, []

    # use mask to create inlier_matches
    
    inlier_matches = [m for m, keep in zip(matches, mask.ravel()) if keep]
    

    return F, inlier_matches

pair_stats = []
for i in range(0, 88):
    img1 = cv2.imread(f"{FRAME_DIR}/frame_{i:04d}.jpg")
    img2 = cv2.imread(f"{FRAME_DIR}/frame_{i+1:04d}.jpg")

    if img1 is None or img2 is None:
        print(f"Could not load frames {i} and {i+1}")
        continue

    kp1, kp2, good_matches = match_frames(img1, img2)

    F, inlier_matches = geometric_filter(kp1, kp2, good_matches)

    if len(good_matches) < 8:
        pair_stats.append(
            {
                "frame_a": f"frame_{i:04d}",
                "frame_b": f"frame_{i+1:04d}",
                "keypoints_a": len(kp1),
                "keypoints_b": len(kp2),
                "lowe_matches": len(good_matches),
                "match_ratio": len(good_matches) / min(len(kp1), len(kp2)),
                "ransac_inliers": 0,
                "inlier_ratio": 0
            }
        )
        continue
    pair_stats.append(
                {
                    "frame_a": f"frame_{i:04d}",
                    "frame_b": f"frame_{i+1:04d}",
                    "keypoints_a": len(kp1),
                    "keypoints_b": len(kp2),
                    "lowe_matches": len(good_matches),
                    "match_ratio": len(good_matches) / min(len(kp1), len(kp2)),
                    "ransac_inliers": len(inlier_matches),
                    "inlier_ratio": len(inlier_matches) / len(good_matches)
                }
            )
    


overlap_df = pd.DataFrame(pair_stats)
overlap_df.to_csv("data/pair_stats.csv", index=False)   
# overlap = len(good_matches) / min(len(kp1), len(kp2))
weakest = overlap_df.sort_values(by="ransac_inliers", ascending=True).head(10)

print(
    weakest[
        [
            "frame_a",
            "frame_b",
            "lowe_matches",
            "ransac_inliers",
            "inlier_ratio"
        ]
    ]
)


lowe = [x["lowe_matches"] for x in pair_stats]
inliers = [x["ransac_inliers"] for x in pair_stats]
ratios = [x["inlier_ratio"] for x in pair_stats]

print("Pairs analysed:", len(pair_stats))

print("Lowe matches:",
      np.min(lowe), np.median(lowe), np.max(lowe))

print("RANSAC inliers:",
      np.min(inliers), np.median(inliers), np.max(inliers))

print("Inlier ratio:",
      np.min(ratios), np.median(ratios), np.max(ratios))

matched_img = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    inlier_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
# cv2.imwrite("output/sift_matches.jpg", matched_img)
# cv2.imwrite("output/inlier_matches.jpg", matched_img)
# cv2.imshow("Matches", matched_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
