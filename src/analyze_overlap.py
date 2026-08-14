import cv2
import numpy as np

FRAME_DIR = "data/chair_001/frames"

img1 = cv2.imread(f"{FRAME_DIR}/frame_0000.jpg")
img2 = cv2.imread(f"{FRAME_DIR}/frame_0001.jpg")

if img1 is None or img2 is None:
    raise ValueError("Could not load frames")

def match_frames(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)



    lowe_ratio = 0.75

    lowe_filtered_matches = []
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

kp1, kp2, good_matches = match_frames(img1, img2)

pts1 = np.float32([
    kp1[m.queryIdx].pt
    for m in good_matches
])

pts2 = np.float32([
    kp2[m.trainIdx].pt
    for m in good_matches
])

F, mask = cv2.findFundamentalMat(
    pts1,
    pts2,
    cv2.FM_RANSAC,
    1.0,
    0.99
)

inlier_matches = [m for m, keep in zip(good_matches, mask.ravel()) if keep]
# print("Keypoints A:", len(kp1))
# print("Keypoints B:", len(kp2))
# print("Good matches:", len(good_matches))

overlap = len(good_matches) / min(len(kp1), len(kp2))

print("Match ratio:", overlap)

print("Lowe matches:", len(good_matches))
print("RANSAC inliers:", len(inlier_matches))

inlier_ratio = len(inlier_matches) / len(good_matches)

print("Geometric inlier ratio:", inlier_ratio)

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
cv2.imwrite("output/inlier_matches.jpg", matched_img)
cv2.imshow("Matches", matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
