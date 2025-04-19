import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from obj_loader import OBJ
from PIL import Image
import tempfile
import time
import webbrowser

st.set_page_config(page_title="3D Marker AR in Streamlit", layout="wide")

st.title("📷 AR Marker Detection with 3D OBJ Model Overlay")

# Set up ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))
marker_length = 0.1

# Load overlay image and model
overlay_image = cv2.imread("RubixCube.png")
model = OBJ("RubixCube.obj")

model_scale = 0.05
model_url_map = {0: "https://example.com/tower", 1: "https://example.com/cube"}
clicked_markers = {}
COOLDOWN_TIME = 2

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Helper for rotating model
def rotate_vertices(vertices, angle_deg, axis='x'):
    angle_rad = np.radians(angle_deg)
    if axis == 'x':
        R = np.array([[1, 0, 0],
                      [0, np.cos(angle_rad), -np.sin(angle_rad)],
                      [0, np.sin(angle_rad),  np.cos(angle_rad)]])
    elif axis == 'y':
        R = np.array([[np.cos(angle_rad), 0, np.sin(angle_rad)],
                      [0, 1, 0],
                      [-np.sin(angle_rad), 0, np.cos(angle_rad)]])
    elif axis == 'z':
        R = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                      [np.sin(angle_rad),  np.cos(angle_rad), 0],
                      [0, 0, 1]])
    else:
        return vertices
    return vertices @ R.T

# Start webcam
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()

st.write("Press `Stop` to release camera when you're done.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        st.warning("Camera not detected.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(gray)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if ids is not None:
        ids = ids.flatten()
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

        for i, marker_id in enumerate(ids):
            rvec = rvecs[i]
            tvec = tvecs[i]

            marker_corners = corners[i][0].astype(np.int32)
            x_min, y_min = np.min(marker_corners, axis=0)
            x_max, y_max = np.max(marker_corners, axis=0)

            if overlay_image is not None:
                h, w = overlay_image.shape[:2]
                dst_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
                matrix, _ = cv2.findHomography(dst_corners, marker_corners)
                warped_img = cv2.warpPerspective(overlay_image, matrix, (frame.shape[1], frame.shape[0]))

                mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                cv2.fillConvexPoly(mask, marker_corners.astype(int), 255)
                mask_inv = cv2.bitwise_not(mask)

                bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
                fg = cv2.bitwise_and(warped_img, warped_img, mask=mask)
                frame = cv2.add(bg, fg)

            tvec = tvec + np.array([0, -0.05, 0], dtype=np.float32)

            rotated_verts = rotate_vertices(model.vertices, 90, axis='x')
            verts = rotated_verts * model_scale

            imgpts, _ = cv2.projectPoints(verts, rvec, tvec, camera_matrix, dist_coeffs)
            imgpts = imgpts.reshape(-1, 2).astype(int)

            for face in model.faces:
                for j in range(len(face)):
                    pt1 = tuple(imgpts[face[j]])
                    pt2 = tuple(imgpts[face[(j + 1) % len(face)]])
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            # Click detection
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    x_finger = int(tip.x * frame.shape[1])
                    y_finger = int(tip.y * frame.shape[0])

                    if x_min <= x_finger <= x_max and y_min <= y_finger <= y_max:
                        current_time = time.time()
                        if marker_id not in clicked_markers or (current_time - clicked_markers[marker_id] > COOLDOWN_TIME):
                            if marker_id in model_url_map:
                                webbrowser.open(model_url_map[marker_id])
                                clicked_markers[marker_id] = current_time

    # Show in Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels="RGB")

# Cleanup
cap.release()
hands.close()
