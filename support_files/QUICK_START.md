# Lane Detection - Quick Start Guide

## What You Have
- **Pretrained Model**: `models/culane_18.pth` (178MB, trained on real roads)
- **Optimized Script**: `run_lane_detection.py` (fast, minimal lag)

## How to Run

### 1. Activate Virtual Environment
```bash
.venv\Scripts\activate
```

### 2. Run Lane Detection

**Live Camera:**
```bash
python run_lane_detection.py
```

**Video File:**
```bash
python run_lane_detection.py --source video.mp4
```

**Save Output:**
```bash
python run_lane_detection.py --source 0 --output result.mp4
```

## Controls
- **'q'** - Quit
- **'s'** - Save current frame
- **SPACE** - Pause/Resume

## Performance Tips
1. **Point camera at ROAD SCENES** - the model is trained on roads
2. **Lower resolution** for faster FPS (edit line 108 in script)
3. **Use GPU** if available (automatic detection)

## Expected Performance
- **CPU**: ~6-8 FPS
- **GPU**: ~30+ FPS

## What Gets Detected
- Up to 4 lanes (colored: Blue, Green, Red, Yellow)
- Works best on highway/road scenes
- May not work well on indoor scenes or non-road images

## Files Kept
- `run_lane_detection.py` - Main optimized script
- `models/culane_18.pth` - Pretrained weights
- Original demo/test/train scripts (if you need them later)

## Removed (Cleanup)
- Test scripts
- Duplicate demos
- Sample videos
- Output images
