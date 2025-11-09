"""
Live Camera Module for Real-time Face Recognition
Streamlit component for webcam access
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

API_BASE_URL = "http://localhost:8000"

class VideoTransformer(VideoTransformerBase):
    """Video transformer for real-time processing"""
    
    def __init__(self):
        self.detection_enabled = False
        self.recognition_enabled = False
        self.last_process_time = 0
        self.process_interval = 0.5  # Process every 0.5 seconds
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        current_time = time.time()
        
        # Process frame at intervals
        if current_time - self.last_process_time > self.process_interval:
            if self.detection_enabled or self.recognition_enabled:
                # Convert to RGB for API
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)
                
                # Save to bytes
                import io
                buf = io.BytesIO()
                pil_img.save(buf, format='JPEG')
                buf.seek(0)
                
                try:
                    if self.recognition_enabled:
                        # Recognition
                        files = {'file': buf}
                        response = requests.post(
                            f"{API_BASE_URL}/api/v1/recognize",
                            files=files,
                            data={'threshold': 0.6},
                            timeout=2
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            if result.get('success'):
                                # Draw results
                                for face in result.get('faces', []):
                                    bbox = face.get('bbox', [])
                                    if len(bbox) == 4:
                                        x, y, w, h = bbox
                                        identity = face.get('identity', 'Unknown')
                                        confidence = face.get('match_confidence', 0)
                                        
                                        # Draw bounding box
                                        color = (0, 255, 0) if identity != 'Unknown' else (0, 0, 255)
                                        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                                        
                                        # Draw label
                                        label = f"{identity} ({confidence:.2%})"
                                        cv2.putText(img, label, (x, y-10),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    elif self.detection_enabled:
                        # Detection only
                        files = {'file': buf}
                        response = requests.post(
                            f"{API_BASE_URL}/api/v1/detect",
                            files=files,
                            timeout=2
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            if result.get('success'):
                                # Draw bounding boxes
                                for face in result.get('faces', []):
                                    bbox = face.get('bbox', [])
                                    if len(bbox) == 4:
                                        x, y, w, h = bbox
                                        confidence = face.get('confidence', 0)
                                        
                                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                        label = f"Face ({confidence:.2%})"
                                        cv2.putText(img, label, (x, y-10),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                except Exception as e:
                    # Silently ignore errors in live feed
                    pass
                
                self.last_process_time = current_time
        
        return img

def show_live_camera():
    """Streamlit page for live camera"""
    st.header("📹 Live Camera Recognition")
    st.markdown("Real-time face detection and recognition using your webcam")
    
    # Controls
    col1, col2 = st.columns(2)
    
    with col1:
        mode = st.radio(
            "Mode:",
            ["Off", "Detection Only", "Full Recognition"],
            horizontal=True
        )
    
    with col2:
        st.info(f"""
        **Current Mode:** {mode}
        - **Off:** Just display camera
        - **Detection:** Find faces
        - **Recognition:** Identify people
        """)
    
    # WebRTC configuration
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Start webcam
    ctx = webrtc_streamer(
        key="face-recognition-live",
        video_transformer_factory=VideoTransformer,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Update transformer settings
    if ctx.video_transformer:
        if mode == "Detection Only":
            ctx.video_transformer.detection_enabled = True
            ctx.video_transformer.recognition_enabled = False
        elif mode == "Full Recognition":
            ctx.video_transformer.detection_enabled = False
            ctx.video_transformer.recognition_enabled = True
        else:
            ctx.video_transformer.detection_enabled = False
            ctx.video_transformer.recognition_enabled = False
    
    st.markdown("---")
    
    # Instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        **Steps:**
        1. Allow camera access when prompted
        2. Select a mode (Detection or Recognition)
        3. Position your face in front of the camera
        4. See real-time results overlaid on video
        
        **Tips:**
        - Ensure good lighting
        - Face the camera directly
        - Stay within 1-2 meters from camera
        - For recognition, add yourself to gallery first
        
        **Performance:**
        - Processes frames every 0.5 seconds
        - Lower interval = faster but more CPU usage
        - Green box = recognized, Red box = unknown
        """)

if __name__ == "__main__":
    show_live_camera()
