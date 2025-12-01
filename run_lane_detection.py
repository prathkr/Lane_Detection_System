"""
Optimized Lane Detection for Camera/Video
- Uses pretrained CULane model
- Optimized for speed with minimal overhead
- Clean visualization
"""

import torch
import cv2
import numpy as np
from model.model import parsingNet
import torchvision.transforms as transforms
from PIL import Image
import scipy.special
import time
import argparse

# Optimized preprocessing - reuse transform
img_transforms = transforms.Compose([
    transforms.Resize((288, 800)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# CULane row anchors
ROW_ANCHOR = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
GRIDING_NUM = 200
CLS_NUM_PER_LANE = 18

# Lane colors (BGR)
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

def process_output(out, img_w, img_h):
    """Fast output processing"""
    out_j = out[0].data.cpu().numpy()
    out_j = out_j[:, ::-1, :]
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    idx = np.arange(GRIDING_NUM) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == GRIDING_NUM] = 0
    
    col_sample = np.linspace(0, 800 - 1, GRIDING_NUM)
    col_sample_w = col_sample[1] - col_sample[0]
    
    lanes = []
    for lane_idx in range(out_j.shape[1]):
        if np.sum(loc[:, lane_idx] != 0) > 2:
            points = []
            for k in range(CLS_NUM_PER_LANE):
                if loc[k, lane_idx] > 0:
                    x = int(loc[k, lane_idx] * col_sample_w * img_w / 800)
                    y = int(img_h * (ROW_ANCHOR[CLS_NUM_PER_LANE-1-k]/288))
                    if 0 <= x < img_w and 0 <= y < img_h:
                        points.append((x, y))
            if len(points) > 1:
                lanes.append(points)
    
    return lanes

def draw_lanes_fast(img, lanes):
    """Optimized lane drawing"""
    for idx, points in enumerate(lanes):
        color = COLORS[idx % 4]
        # Draw thick lines
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i+1], color, 3, cv2.LINE_AA)
        # Draw points
        for point in points:
            cv2.circle(img, point, 4, color, -1)
    return img

def run_detection(source=0, show_fps=True, save_output=None):
    """Main detection loop - optimized"""
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device.upper()}")
    
    # Load model
    print("Loading model...")
    net = parsingNet(pretrained=False, backbone='18', 
                     cls_dim=(GRIDING_NUM+1, CLS_NUM_PER_LANE, 4), 
                     use_aux=False).to(device)
    
    # Load weights
    state_dict = torch.load('models/culane_18.pth', map_location=device)['model']
    compatible_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()
    
    # Disable gradient computation for speed
    torch.set_grad_enabled(False)
    
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    print("✓ Model ready\n")
    
    # Open video source
    if isinstance(source, int):
        print(f"Opening camera {source}...")
    else:
        print(f"Opening video: {source}")
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("ERROR: Cannot open video source")
        return
    
    # Set camera properties for speed
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Resolution: {width}x{height} @ {fps}fps")
    
    # Video writer
    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
        print(f"Saving to: {save_output}")
    
    print("\nControls: 'q'=Quit | 's'=Save frame | SPACE=Pause")
    print("="*50 + "\n")
    
    frame_count = 0
    fps_list = []
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                start_time = time.time()
                
                # Preprocess
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                x = img_transforms(img_pil).unsqueeze(0).to(device)
                
                # Inference
                out = net(x)
                
                # Process and draw
                lanes = process_output(out, width, height)
                result = draw_lanes_fast(frame, lanes)
                
                # Calculate FPS
                elapsed = time.time() - start_time
                current_fps = 1.0 / elapsed if elapsed > 0 else 0
                fps_list.append(current_fps)
                if len(fps_list) > 30:
                    fps_list.pop(0)
                avg_fps = np.mean(fps_list)
                
                # Add minimal overlay
                if show_fps:
                    cv2.putText(result, f"FPS: {avg_fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(result, f"Lanes: {len(lanes)}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Write to file
                if writer:
                    writer.write(result)
                
                # Display
                cv2.imshow('Lane Detection', result)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"lane_frame_{frame_count}.jpg"
                cv2.imwrite(filename, result)
                print(f"Saved: {filename}")
            elif key == ord(' '):
                paused = not paused
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        if frame_count > 0:
            print(f"\nProcessed: {frame_count} frames")
            print(f"Average FPS: {np.mean(fps_list):.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fast Lane Detection')
    parser.add_argument('--source', default='0', help='Camera index or video file')
    parser.add_argument('--output', default=None, help='Save output video')
    parser.add_argument('--no-fps', action='store_true', help='Hide FPS counter')
    
    args = parser.parse_args()
    
    # Convert source to int if camera index
    try:
        source = int(args.source)
    except:
        source = args.source
    
    run_detection(source, show_fps=not args.no_fps, save_output=args.output)
