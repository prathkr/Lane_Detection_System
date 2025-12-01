import streamlit as st
import cv2
import torch
import numpy as np
from model.model import parsingNet
import torchvision.transforms as transforms
from PIL import Image
import scipy.special
import time
import tempfile
import os

# --- Constants & Configuration ---
ROW_ANCHOR = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
GRIDING_NUM = 200
CLS_NUM_PER_LANE = 18
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

# Image transforms
img_transforms = transforms.Compose([
    transforms.Resize((288, 800)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# --- Helper Functions ---
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

@st.cache_resource
def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = parsingNet(pretrained=False, backbone='18', 
                     cls_dim=(GRIDING_NUM+1, CLS_NUM_PER_LANE, 4), 
                     use_aux=False).to(device)
    
    # Load weights
    # Assuming the model path is relative to the script
    model_path = 'models/culane_18.pth'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return None, device

    state_dict = torch.load(model_path, map_location=device)['model']
    compatible_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()
    
    # Disable gradient computation for speed
    torch.set_grad_enabled(False)
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        
    return net, device

# --- Main App ---
def main():
    st.set_page_config(page_title="Ultra Fast Lane Detection", layout="wide")
    
    st.title("🛣️ Ultra Fast Lane Detection")
    st.markdown("Upload a video or use your webcam to detect lanes in real-time.")

    # Sidebar
    st.sidebar.header("Settings")
    source_type = st.sidebar.radio("Select Source", ["Video File", "Webcam"])
    
    show_fps = st.sidebar.checkbox("Show FPS", value=True)
    confidence_threshold = st.sidebar.slider("Confidence Threshold (Visual)", 0.0, 1.0, 0.5) # Placeholder for future use if needed

    # Load Model
    with st.spinner("Loading Model..."):
        net, device = load_model()
    
    if net is None:
        return

    st.sidebar.success(f"Model loaded on {device.upper()}")

    # Input Handling
    cap = None
    temp_file_path = None

    if source_type == "Video File":
        uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            temp_file_path = tfile.name
            cap = cv2.VideoCapture(temp_file_path)
    else:
        if st.button("Start Webcam"):
            cap = cv2.VideoCapture(0)
            # Set camera properties for speed
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Processing Loop
    if cap is not None and cap.isOpened():
        st_frame = st.empty()
        stop_button = st.button("Stop")
        
        fps_list = []
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.write("End of video or cannot read frame.")
                break
            
            start_time = time.time()

            # Preprocess
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            x = img_transforms(img_pil).unsqueeze(0).to(device)

            # Inference
            out = net(x)

            # Process and draw
            lanes = process_output(out, frame.shape[1], frame.shape[0])
            result = draw_lanes_fast(frame.copy(), lanes) # Draw on copy to keep original clean if needed

            # Calculate FPS
            elapsed = time.time() - start_time
            current_fps = 1.0 / elapsed if elapsed > 0 else 0
            fps_list.append(current_fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = np.mean(fps_list)

            # Overlay FPS
            if show_fps:
                cv2.putText(result, f"FPS: {avg_fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Display in Streamlit
            # Convert BGR to RGB for Streamlit
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            st_frame.image(result_rgb, channels="RGB", use_column_width=True)

        cap.release()
        if temp_file_path:
            os.remove(temp_file_path)

if __name__ == "__main__":
    main()
