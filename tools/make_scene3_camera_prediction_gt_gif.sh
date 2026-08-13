#!/usr/bin/env bash
set -euo pipefail

# Camera | RadarOcc prediction | GT comparison for scene 3.
# Prediction and GT panels are rotated 90 degrees counter-clockwise.
# Differently numbered files are synchronized by natural-sorted ordinal order.

cd /home/user1/projects/RadarOcc

EXP="/home/user1/projects/RadarOcc/work_dirs/radarocc_small_fp32_idfix_timealign_v2"
VIS="$EXP/visualization_epoch4_test"
SCENE=3

PRED_DIR="$VIS/$SCENE"
GT_DIR="/home/user1/projects/RadarOcc/data/K-RadarOcc/train/3/semantic_occupancy_gt_fov"
CAM_DIR="/home/user1/projects/RadarOcc/data/K-Radar-RGB/K-Radar/K-Radar-RGB/3/images_rb_switched"
RENDER_SCRIPT="/home/user1/projects/RadarOcc/tools/render_occ_monoscene_video.py"

START=0
COUNT=100
FPS=10

# Existing renderer output size.
RENDER_WIDTH=960
RENDER_HEIGHT=540

# Crop the 3-D viewport from the 960x540 renderer frame before rotation.
# These defaults match the screenshot supplied in the conversation.
OCC_CROP_W=300
OCC_CROP_H=420
OCC_CROP_X=330
OCC_CROP_Y=60

OUT_DIR="$VIS/scene_${SCENE}_camera_pred_gt"
mkdir -p "$OUT_DIR"
TAG=$(printf '%03d' "$START")

PRED_LIST="$OUT_DIR/prediction_files_${TAG}.txt"
GT_LIST="$OUT_DIR/gt_files_${TAG}.txt"
CAM_LIST="$OUT_DIR/camera_files_${TAG}.txt"
SYNC_MANIFEST="$OUT_DIR/sync_manifest_${TAG}.tsv"
CAM_CONCAT="$OUT_DIR/camera_concat_${TAG}.txt"

PRED_SUBSET="$OUT_DIR/pred_subset_${TAG}"
GT_SUBSET="$OUT_DIR/gt_subset_${TAG}"

PRED_MP4="$OUT_DIR/prediction_${TAG}.mp4"
GT_MP4="$OUT_DIR/ground_truth_${TAG}.mp4"
CAM_MP4="$OUT_DIR/camera_${TAG}.mp4"
COMBINED_MP4="$OUT_DIR/camera_prediction_gt_${TAG}.mp4"
COMBINED_GIF="$OUT_DIR/camera_prediction_gt_${TAG}.gif"

FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

for cmd in ffmpeg ffprobe xvfb-run python find sort sed awk paste; do
    command -v "$cmd" >/dev/null 2>&1 || {
        echo "Missing command: $cmd" >&2
        exit 1
    }
done

for dir in "$PRED_DIR" "$GT_DIR" "$CAM_DIR"; do
    [[ -d "$dir" ]] || {
        echo "Directory does not exist: $dir" >&2
        exit 1
    }
done
[[ -f "$RENDER_SCRIPT" ]] || {
    echo "Renderer does not exist: $RENDER_SCRIPT" >&2
    exit 1
}

FIRST=$((START + 1))
LAST=$((START + COUNT))

# Natural sorting is deliberate: camera/radar file IDs differ, but the user
# confirmed that the streams are one-to-one in chronological order.
find "$PRED_DIR" -maxdepth 1 -type f -name '*.npy' -printf '%p\n' \
    | sort -V | sed -n "${FIRST},${LAST}p" > "$PRED_LIST"

find "$GT_DIR" -maxdepth 1 -type f -name '*.npy' -printf '%p\n' \
    | sort -V | sed -n "${FIRST},${LAST}p" > "$GT_LIST"

find "$CAM_DIR" -maxdepth 1 -type f \
    \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' \) \
    -printf '%p\n' | sort -V | sed -n "${FIRST},${LAST}p" > "$CAM_LIST"

PRED_COUNT=$(wc -l < "$PRED_LIST")
GT_COUNT=$(wc -l < "$GT_LIST")
CAM_COUNT=$(wc -l < "$CAM_LIST")

echo "Prediction frames: $PRED_COUNT"
echo "GT frames:         $GT_COUNT"
echo "Camera frames:     $CAM_COUNT"

if [[ "$PRED_COUNT" -ne "$COUNT" || "$GT_COUNT" -ne "$COUNT" || "$CAM_COUNT" -ne "$COUNT" ]]; then
    echo "Expected exactly $COUNT files from every source." >&2
    echo "Check START/COUNT and the three input directories." >&2
    exit 1
fi

# Write the exact synchronization table for manual verification.
{
    printf 'ordinal\tprediction\tground_truth\tcamera\n'
    paste "$PRED_LIST" "$GT_LIST" "$CAM_LIST" \
        | awk -F '\t' -v start="$START" '{
            printf "%d\t%s\t%s\t%s\n", start + NR - 1, $1, $2, $3
        }'
} > "$SYNC_MANIFEST"

echo "Synchronization table: $SYNC_MANIFEST"

# Create ordered subset directories. This prevents the renderer from using a
# different filename ordering and makes prediction/GT frame 0 correspond exactly.
rm -rf "$PRED_SUBSET" "$GT_SUBSET"
mkdir -p "$PRED_SUBSET" "$GT_SUBSET"

idx=0
while IFS= read -r file; do
    ln -s "$file" "$PRED_SUBSET/$(printf '%06d.npy' "$idx")"
    idx=$((idx + 1))
done < "$PRED_LIST"

idx=0
while IFS= read -r file; do
    ln -s "$file" "$GT_SUBSET/$(printf '%06d.npy' "$idx")"
    idx=$((idx + 1))
done < "$GT_LIST"

render_occ_video() {
    local input_dir=$1
    local output_file=$2

    if ffprobe -v error "$output_file" >/dev/null 2>&1; then
        echo "Reuse completed video: $output_file"
        return
    fi

    rm -f "$output_file"
    unset QT_QPA_PLATFORM

    xvfb-run -a \
        -s "-screen 0 1280x720x24" \
        env \
          LIBGL_ALWAYS_SOFTWARE=1 \
          ETS_TOOLKIT=qt \
          QT_API=pyqt5 \
        python "$RENDER_SCRIPT" \
          "$input_dir" \
          --output "$output_file" \
          --fps "$FPS" \
          --start 0 \
          --max-frames "$COUNT" \
          --width "$RENDER_WIDTH" \
          --height "$RENDER_HEIGHT"

    ffprobe -v error "$output_file" >/dev/null 2>&1 || {
        echo "Invalid rendered video: $output_file" >&2
        exit 1
    }
}

echo
 echo "Rendering prediction..."
render_occ_video "$PRED_SUBSET" "$PRED_MP4"

echo
 echo "Rendering ground truth with the same renderer..."
render_occ_video "$GT_SUBSET" "$GT_MP4"

# Build a 10-fps camera video from the same chronological positions.
rm -f "$CAM_CONCAT"
FRAME_DURATION=$(awk -v fps="$FPS" 'BEGIN {printf "%.10f", 1.0/fps}')
while IFS= read -r image; do
    printf "file '%s'\n" "$image" >> "$CAM_CONCAT"
    printf "duration %s\n" "$FRAME_DURATION" >> "$CAM_CONCAT"
done < "$CAM_LIST"
# Repeat the final file so the concat demuxer honors the final duration.
printf "file '%s'\n" "$(tail -n 1 "$CAM_LIST")" >> "$CAM_CONCAT"

ffmpeg -y \
    -f concat -safe 0 -i "$CAM_CONCAT" \
    -vf "fps=${FPS},format=yuv420p" \
    -an -c:v libx264 -crf 18 -preset medium \
    "$CAM_MP4"

# transpose=2 means exactly 90 degrees counter-clockwise.
# Only the cropped 3-D viewport is rotated; the old title is cropped away and
# replaced with a normal horizontal label.
if [[ -f "$FONT" ]]; then
    CAM_LABEL="drawtext=fontfile=${FONT}:text='Camera':fontcolor=white:fontsize=28:x=(w-text_w)/2:y=14:box=1:boxcolor=black@0.60:boxborderw=8"
    PRED_LABEL="drawtext=fontfile=${FONT}:text='Prediction':fontcolor=black:fontsize=28:x=(w-text_w)/2:y=16"
    GT_LABEL="drawtext=fontfile=${FONT}:text='Ground Truth':fontcolor=black:fontsize=28:x=(w-text_w)/2:y=16"
else
    CAM_LABEL="drawtext=text='Camera':fontcolor=white:fontsize=28:x=(w-text_w)/2:y=14:box=1:boxcolor=black@0.60:boxborderw=8"
    PRED_LABEL="drawtext=text='Prediction':fontcolor=black:fontsize=28:x=(w-text_w)/2:y=16"
    GT_LABEL="drawtext=text='Ground Truth':fontcolor=black:fontsize=28:x=(w-text_w)/2:y=16"
fi

echo
 echo "Composing Camera | Prediction | Ground Truth..."
ffmpeg -y \
    -i "$CAM_MP4" \
    -i "$PRED_MP4" \
    -i "$GT_MP4" \
    -filter_complex "\
[0:v]fps=${FPS},scale=780:650:force_original_aspect_ratio=decrease,pad=800:720:(ow-iw)/2:(oh-ih)/2:black,${CAM_LABEL},setpts=PTS-STARTPTS[cam];\
[1:v]fps=${FPS},crop=${OCC_CROP_W}:${OCC_CROP_H}:${OCC_CROP_X}:${OCC_CROP_Y},transpose=2,scale=480:-2:flags=lanczos,pad=500:720:(ow-iw)/2:(oh-ih)/2:white,${PRED_LABEL},setpts=PTS-STARTPTS[pred];\
[2:v]fps=${FPS},crop=${OCC_CROP_W}:${OCC_CROP_H}:${OCC_CROP_X}:${OCC_CROP_Y},transpose=2,scale=480:-2:flags=lanczos,pad=500:720:(ow-iw)/2:(oh-ih)/2:white,${GT_LABEL},setpts=PTS-STARTPTS[gt];\
[cam][pred][gt]hstack=inputs=3,format=yuv420p[out]" \
    -map '[out]' -an -r "$FPS" -shortest \
    -c:v libx264 -crf 18 -preset medium \
    "$COMBINED_MP4"

ffprobe -v error "$COMBINED_MP4" >/dev/null 2>&1 || {
    echo "Invalid combined MP4: $COMBINED_MP4" >&2
    exit 1
}

# Keep the real data rate: 10 frames = 1 second.
ffmpeg -y \
    -i "$COMBINED_MP4" \
    -filter_complex "fps=${FPS},scale=1800:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=diff[p];[s1][p]paletteuse=dither=sierra2_4a" \
    -loop 0 \
    "$COMBINED_GIF"

echo
echo "Finished"
echo "MP4:     $COMBINED_MP4"
echo "GIF:     $COMBINED_GIF"
echo "Mapping: $SYNC_MANIFEST"
