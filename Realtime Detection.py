pip install playsound
pip install pygame

from flask import Flask, request, render_template, redirect, url_for, jsonify, Response
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os
from flask_ngrok import run_with_ngrok
import threading
import time
import pygame

app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when the app is run

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Load CNN-LSTM model for gait detection
cnn_lstm_model = load_model('./cnn_lstm.h5')  # Update with your model weights path

# Initialize pygame mixer
pygame.mixer.init()

# Load the sound file once at the beginning
fall_sound = pygame.mixer.Sound('./static/fall_audio.wav')
movement_sound = pygame.mixer.Sound('./static/movement.mp3')

# Global variable to control the real-time detection thread
is_running = False

# Define constants
FRAME_SEQUENCE_LENGTH = 30
FALL_THRESHOLD = 0.2  # Threshold for hip Y-coordinate change
FALL_CONFIRMATIONS_REQUIRED = 5  # Confirmations required for fall detection

# Calibration variables
calibrated = False
calibration_frames = []
calibration_frame_count = 30  # Number of frames to use for calibration

# Function to preprocess frame for CNN-LSTM model
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (64, 64))
    normalized_frame = resized_frame / 255.0  # Normalize to [0, 1]
    return normalized_frame

# Function to detect gait using CNN-LSTM model
def detect_gait_with_cnn_lstm(frames):
    processed_frames = np.array([preprocess_frame(frame) for frame in frames])
    processed_frames = np.expand_dims(processed_frames, axis=0)
    predictions = cnn_lstm_model.predict(processed_frames)
    return predictions

# Function to detect action based on the results from MediaPipe
def detect_action(results, prev_results):
    if results.pose_landmarks is None:
        return 'Unknown'

    landmarks = results.pose_landmarks.landmark
    if prev_results is None or prev_results.pose_landmarks is None:
        return 'Unknown'

    # Action detection logic
    if detect_laying_down(results):
        return 'Laying Down'
    if detect_sitting_down(results, prev_results):
        return 'Sitting Down'
    if detect_sitting(results):
        return 'Sitting'
    if detect_standing_up(results, prev_results):
        return 'Standing Up'
    if detect_standing(results):
        return 'Standing'
    if detect_walking(results, prev_results):
        return 'Walking'

    return 'Unknown'

# Define action detection functions
def detect_laying_down(results):
    landmarks = results.pose_landmarks.landmark
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    shoulder_hip_diff = abs(left_shoulder.y - left_hip.y) + abs(right_shoulder.y - right_hip.y)
    return shoulder_hip_diff < 0.05

def detect_sitting_down(results, prev_results):
    landmarks = results.pose_landmarks.landmark
    prev_landmarks = prev_results.pose_landmarks.landmark
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    hip_knee_ratio = (left_hip.y + right_hip.y) / (left_knee.y + right_knee.y)
    prev_hip_knee_ratio = (
        prev_landmarks[mp_pose.PoseLandmark.LEFT_HIP].y +
        prev_landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
    ) / (
        prev_landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y +
        prev_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y
    )
    return prev_hip_knee_ratio > 1 and hip_knee_ratio < 1

def detect_sitting(results):
    landmarks = results.pose_landmarks.landmark
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    hip_knee_ratio = (left_hip.y + right_hip.y) / (left_knee.y + right_knee.y)
    return 0.95 <= hip_knee_ratio <= 1.05 and left_hip.y > 0.5 and right_hip.y > 0.5

def detect_standing_up(results, prev_results):
    landmarks = results.pose_landmarks.landmark
    prev_landmarks = prev_results.pose_landmarks.landmark
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    hip_knee_ratio = (left_hip.y + right_hip.y) / (left_knee.y + right_knee.y)
    prev_hip_knee_ratio = (prev_landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + prev_landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / \
                          (prev_landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y + prev_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y)
    return prev_hip_knee_ratio < 1 and hip_knee_ratio > 1

def detect_standing(results):
    landmarks = results.pose_landmarks.landmark
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    return left_hip.y < 0.5 and right_hip.y < 0.5 and left_knee.y > 0.5 and right_knee.y > 0.5

def detect_walking(results, prev_results):
    landmarks = results.pose_landmarks.landmark
    prev_landmarks = prev_results.pose_landmarks.landmark
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    prev_left_ankle = prev_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    prev_right_ankle = prev_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    ankle_movement = (left_ankle.x - prev_left_ankle.x) * (right_ankle.x - prev_right_ankle.x)
    return ankle_movement < 0

def calibrate():
    global calibrated, calibration_frames
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Couldn't open video capture.")
        return
    
    calibration_frames = []
    frame_count = 0

    while frame_count < calibration_frame_count:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            calibration_frames.append(frame)
            frame_count += 1

    cap.release()
    calibrated = True
    print("Calibration completed.")

def detect_gait_realtime():
    global is_running, calibrated
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Couldn't open video capture.")
        return
    
    frame_sequence = []
    prev_results = None
    prev_hip_y = None
    fall_confirmations = 0
    fallen = False
    human_in_view = False  # Flag to check if human is in view
    no_human_detected_count = 0  # Counter for no human detected frames

    is_running = True  

    while is_running:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            no_human_detected_count = 0  # Reset counter when human is detected
            if not human_in_view:  # Human was out of view but is now in view
                print("Human returned to view, stopping movement sound")
                movement_sound.stop()  # Stop the movement sound
            human_in_view = True  # Human is in view
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            action_label = detect_action(results, prev_results) if prev_results else 'Unknown'
            prev_results = results

            frame_sequence.append(frame)
            if len(frame_sequence) == FRAME_SEQUENCE_LENGTH:
                predictions = detect_gait_with_cnn_lstm(frame_sequence)
                if np.mean(predictions) > 0.5:
                    fall_confirmations += 1
                    if fall_confirmations >= FALL_CONFIRMATIONS_REQUIRED:
                        fallen = True
                        cv2.putText(frame, "FALL DETECTED", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        fall_sound.play()  # Play the alert sound when a fall is detected
                else:
                    fall_confirmations = 0  # Reset confirmations if fall is not detected
                frame_sequence.pop(0)  

            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            if left_hip and right_hip:
                hip_y = (left_hip.y + right_hip.y) / 2  # Average hip Y-coordinate

                if prev_hip_y is not None:
                    hip_acceleration = (hip_y - prev_hip_y) / (1 / 30)  # Assuming 30 FPS
                    if hip_acceleration > FALL_THRESHOLD:
                        fall_confirmations += 1
                        if fall_confirmations >= FALL_CONFIRMATIONS_REQUIRED:
                            fallen = True
                            cv2.putText(frame, "FALL DETECTED", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            fall_sound.play()  # Play the alert sound when a fall is detected
                    else:
                        fall_confirmations = 0  # Reset confirmations if fall is not detected
                prev_hip_y = hip_y

            cv2.putText(frame, f"Action: {action_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        else:
            no_human_detected_count += 1
            if no_human_detected_count > 10:  # If no human detected for more than 10 frames
                if human_in_view:  # Human was in view but is now out of view
                    print("Human out of view, playing movement sound")
                    movement_sound.play(-1)  # Play the movement sound in a loop
                    human_in_view = False
                else:
                    print("Human still out of view, continuing movement sound")

        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release()
    is_running = False  # Ensure the flag is set to False when the thread exits

    # Stop the movement sound when the detection stops
    movement_sound.stop()

@app.route('/')
def realtime():
    return render_template('realtime.html')

@app.route('/start', methods=['POST'])
def start_realtime_video():
    global is_running, calibrated
    if not is_running:
        if not calibrated:
            calibrate()
        threading.Thread(target=detect_gait_realtime).start()
        print("Real-time gait detection started...")
        return jsonify({'message': 'Real-time gait detection started...'})
    else:
        print("Real-time gait detection is already running")
        return jsonify({'error': 'Real-time gait detection is already running'}), 400
    
@app.route('/stop', methods=['POST'])
def stop_realtime_video():
    global is_running
    is_running = False
    print("Real-time gait detection stopped")
    movement_sound.stop()  # Ensure the movement sound stops
    return jsonify({'message': 'Real-time gait detection stopped'})

@app.route('/video_feed')
def video_feed():
    return Response(detect_gait_realtime(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/exit')
def exit_system():
    return "System Exited", 200

if __name__ == "__main__":
    app.run()

    #### REALTIME
    #### SETTLE
    #15/11
