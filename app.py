from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import pyautogui

app = Flask(__name__)

# Khởi tạo Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Kích thước màn hình
screen_w, screen_h = pyautogui.size()

# Mở webcam
cap = cv2.VideoCapture(0)

@app.route("/")
def home():
    return render_template("home.html")

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Lật ảnh
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Chuyển ảnh sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                index_finger = hand_landmarks.landmark[8]  # Ngón trỏ
                x, y = int(index_finger.x * w), int(index_finger.y * h)

                # Chuyển đổi tọa độ từ webcam sang màn hình máy tính
                mouse_x = np.interp(x, [0, w], [0, screen_w])
                mouse_y = np.interp(y, [0, h], [0, screen_h])

                # Điều khiển chuột
                pyautogui.moveTo(mouse_x, mouse_y)

                # Vẽ bàn tay lên ảnh
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Mã hóa ảnh thành JPEG để stream
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
  app.run(debug=True)