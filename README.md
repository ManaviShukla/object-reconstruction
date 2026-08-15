1. Run src/extract_frame.py - to extract images from video on fixed time intervals
2. Run analyze_overlap.py to get image pair statistics and pair wise quality using RANSAC
3. Run prepare_reconstruction.py to filter quality images with sharpness.
4. Post this we move to colmap to create the keypoints + SIFT descriptors for us.
    I installed COLMAP binary through > 
    ```bash
    vcpkg install colmap[cuda,tests]:x64-windows
    ```
5. once installed, set the path and run the command >
     ```bash
    colmap feature_extractor `
    --database_path "data/chair_001/reconstruction/database.db" `
    --image_path "data/chair_001/colmap_input" `
    --ImageReader.single_camera 1
    ```
6. Now we do feature matching. We use exhaustive matcher because our sample size is small. For a bigger video dataset we would use sequential matching or spatial matching when u have location priors like GPS>
    ```bash
    colmap exhaustive_matcher `
    --database_path "data/chair_001/reconstruction/database.db"
    ```


7.  Now come the triangulation -we use the pairwise correspondences to estimate camera poses and build sparse 3D points.>
    ```bash
    colmap mapper `
    --database_path "data/chair_001/reconstruction/database.db" `
    --image_path "data/chair_001/colmap_input" `
    --output_path "data/chair_001/reconstruction/sparse"
    ```

    Ref: https://colmap.github.io/tutorial.html

    Note: SFM might fail here if we dont have our camera's focal length in the meta data or because it cannot find a good image pair, due to weak matching or unknown camera calibration, but we can pass a manually selected pair.
    