# Create COLMAP folders
mkdir -p data/chair_001/colmap_input
mkdir -p data/chair_001/colmap/sparse

# Clear any previous COLMAP input
rm -f data/chair_001/colmap_input/*.jpg

# Copy frames with sharpness >= 15
awk -F',' '
NR==1 {
    for (i=1; i<=NF; i++) {
        if ($i=="frame_id") frame_col=i
        if ($i=="sharpness_score" || $i=="sharpness") sharp_col=i
    }
    next
}
$sharp_col >= 15 {
    printf "data/chair_001/frames/frame_%04d.jpg\n", $frame_col
}
' data/chair_001/frames/metadata.csv |
while read frame; do
    cp "$frame" data/chair_001/colmap_input/
done

echo "Frames selected:"
find data/chair_001/colmap_input -name "*.jpg" | wc -l