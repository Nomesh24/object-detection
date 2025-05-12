import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
import os
from datetime import datetime

# ✅ MUST be the first Streamlit command
st.set_page_config(page_title="YOLOv4 Object Detection", layout="centered")

# ---------------------- CSS for Background -------------------------
page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1521790360285-75b1e66b8fb4");
background-size: cover;
background-repeat: no-repeat;
background-attachment: fixed;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# ------------------- Load YOLOv4 Model -----------------------------
net = cv2.dnn.readNet("./configv4/yolov4.weights", "./configv4/yolov4.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# Load class names
with open("./configv4/coco.names", 'r') as f:
    classes = f.read().splitlines()

# Get output layers
layer_names = net.getLayerNames()
try:
    output_layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except IndexError:
    output_layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Session State Initialization
if "history" not in st.session_state:
    st.session_state.history = []
if "live_running" not in st.session_state:
    st.session_state.live_running = False
if "webcam_started" not in st.session_state:
    st.session_state.webcam_started = False

# ---------------------- Detection Functions ------------------------
def draw_bounding_box(img, class_id, confidence, x, y, w, h):
    label = f"{classes[class_id]}: {confidence:.2f}"
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def detect_objects(frame, target_class=None):
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layer_names)

    h, w = frame.shape[:2]
    boxes, confidences, classIDs = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - width / 2)
                y = int(centerY - height / 2)
                label = classes[classID]
                if target_class is None or label.lower() == target_class.lower():
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected_labels = []

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w_box, h_box = boxes[i]
            draw_bounding_box(frame, classIDs[i], confidences[i], x, y, w_box, h_box)
            label = classes[classIDs[i]]
            detected_labels.append(label)
    return frame, detected_labels

# ---------------------- Streamlit UI -------------------------------
st.title("🔍 YOLOv4 Object Detection")
st.markdown("Detect objects in an image or from your webcam using YOLOv4.")

option = st.radio("Choose input mode:", ("Image Upload", "Webcam (Live Demo)", "Search Specific Object in Webcam"))

# --------------------- Image Upload Mode ---------------------------
if option == "Image Upload":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "tiff"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Objects"):
            output, detected = detect_objects(frame)
            st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption="Detected Image", use_column_width=True)
            if detected:
                st.success(f"Detected: {', '.join(detected)}")
                st.session_state.history.extend(detected)

                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"upload_detected_{timestamp}.jpg"
                save_path = os.path.join("saved_frames", filename)
                os.makedirs("saved_frames", exist_ok=True)
                cv2.imwrite(save_path, output)
                st.markdown(f"[📷 Download Detected Image]({save_path})", unsafe_allow_html=True)
            else:
                st.warning("No objects detected.")

# --------------------- Webcam Live Mode ----------------------------
elif option == "Webcam (Live Demo)":
    FRAME_WINDOW = st.image([])
    start_button = st.button("Start Webcam")
    detect_button = st.button("Detect Objects (Stop Feed and Save Frame)")
    exit_button = st.button("Exit Webcam")

    if start_button and not st.session_state.webcam_started:
        st.session_state.live_running = True
        st.session_state.webcam_started = True
        st.session_state.cap = cv2.VideoCapture(0)

    if detect_button and st.session_state.webcam_started:
        st.session_state.live_running = False

    if exit_button and st.session_state.webcam_started:
        st.session_state.cap.release()
        st.session_state.webcam_started = False
        st.info("Webcam exited.")

    if st.session_state.webcam_started and st.session_state.live_running:
        cap = st.session_state.cap
        ret, frame = cap.read()
        if ret:
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        time.sleep(0.1)

    elif st.session_state.webcam_started and not st.session_state.live_running:
        cap = st.session_state.cap
        ret, frame = cap.read()
        if ret:
            output, detected = detect_objects(frame)
            FRAME_WINDOW.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            if detected:
                st.success(f"Detected: {', '.join(detected)}")
                st.session_state.history.extend(detected)

                # Save the detection frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"webcam_detected_{timestamp}.jpg"
                save_path = os.path.join("saved_frames", filename)
                os.makedirs("saved_frames", exist_ok=True)
                cv2.imwrite(save_path, output)
                st.markdown(f"[📷 Download Detected Frame]({save_path})", unsafe_allow_html=True)
            else:
                st.warning("No objects detected in final frame.")
        st.session_state.live_running = None

# ---------------- Search Specific Object ---------------------------
elif option == "Search Specific Object in Webcam":
    target_object = st.text_input("Enter the object name to search (e.g., person, dog):")
    search = st.button("Search in Webcam Frame")
    FRAME_WINDOW = st.image([])

    if search and target_object:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Webcam not accessible.")
        else:
            ret, frame = cap.read()
            if ret:
                output, detected = detect_objects(frame, target_class=target_object)
                FRAME_WINDOW.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
                if target_object.lower() in [d.lower() for d in detected]:
                    st.success(f"'{target_object}' detected.")
                    st.session_state.history.append(target_object)

                    # Save frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"specific_search_{timestamp}.jpg"
                    save_path = os.path.join("saved_frames", filename)
                    os.makedirs("saved_frames", exist_ok=True)
                    cv2.imwrite(save_path, output)
                    st.markdown(f"[📷 Download Detected Frame]({save_path})", unsafe_allow_html=True)
                else:
                    st.warning(f"'{target_object}' not found.")
            cap.release()

# ---------------------- Detection History --------------------------
if st.session_state.history:
    st.subheader("📜 Detection History")
    st.write(", ".join(st.session_state.history))
