## Capture / Keyframe Selection
- [ ] Replace fixed 0.6s sampling with adaptive keyframe selection
  - Reject blurry frames
  - Detect overly redundant frames
  - Detect correspondence breaks using SIFT + geometric inliers
  - Search intermediate video frames when a correspondence break occurs
  - Eventually provide capture feedback such as "move slower" / "need more coverage"