from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os
from flask_ngrok import run_with_ngrok
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
import time
import requests

app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when the app is run

# Directly assign the new API key
groq_api_key = "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"

# Initialize Groq API with the new key
llm = ChatGroq(temperature=0.5, groq_api_key=groq_api_key, model_name="llama3-8b-8192")

# Define prompt templates for video analysis feedback
insight_prompt_template = PromptTemplate(
    template="""
    You are a video analysis reviewer. Based on the provided metrics, generate concise insights for the video.

    Fall Detection: {fall_detection}
    Action Sequence Analysis: {action_sequence_analysis}
    Fall Risk Analysis: {fall_risk_analysis}
    Processed Actions: {processed_actions}
    Fall Incidents: {fall_incidents}

    Provide concise insights on potential fall risks and overall action sequence. Format the response as a paragraph.
    """,
    input_variables=["fall_detection", "action_sequence_analysis", "fall_risk_analysis", "processed_actions", "fall_incidents"]
)

recommendation_prompt_template = PromptTemplate(
    template="""
    You are a video analysis reviewer. Based on the provided metrics, generate concise recommendations for the video.

    Fall Detection: {fall_detection}
    Action Sequence Analysis: {action_sequence_analysis}
    Fall Risk Analysis: {fall_risk_analysis}
    Processed Actions: {processed_actions}
    Fall Incidents: {fall_incidents}

    Provide concise recommendations to mitigate potential fall risks and improve overall action sequence. Format the response as a paragraph.
    """,
    input_variables=["fall_detection", "action_sequence_analysis", "fall_risk_analysis", "processed_actions", "fall_incidents"]
)

summary_prompt_template = PromptTemplate(
    template="""
    You are a video analysis reviewer. Based on the provided metrics, generate a concise summary for the video.

    Fall Detection: {fall_detection}
    Action Sequence Analysis: {action_sequence_analysis}
    Fall Risk Analysis: {fall_risk_analysis}
    Processed Actions: {processed_actions}
    Fall Incidents: {fall_incidents}

    Provide a concise summary of the video analysis. Format the response as a paragraph.
    """,
    input_variables=["fall_detection", "action_sequence_analysis", "fall_risk_analysis", "processed_actions", "fall_incidents"]
)

output_parser = StrOutputParser()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Load CNN-LSTM model for gait detection
cnn_lstm_model = load_model('./cnn_lstm.h5')  # Update with your model weights path

# Define constants
FRAME_SEQUENCE_LENGTH = 30
ACTION_LABELS = {
    0: 'Walking',
    1: 'Standing',
    2: 'Sitting',
    3: 'Sitting Down',
    4: 'Standing Up',
    5: 'Laying Down',
    6: 'Unknown'
}

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    angle = np.degrees(np.arctan2(c.y - b.y, c.x - b.x) - np.arctan2(a.y - b.y, a.x - b.x))
    angle = angle + 360 if angle < 0 else angle
    return angle

# Function to preprocess frame for CNN-LSTM model
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (64, 64))
    normalized_frame = resized_frame / 255.0  # Normalize to [0, 1]
    return normalized_frame  # Return the processed frame without batch dimension

# Function to detect gait using CNN-LSTM model
def detect_gait_with_cnn_lstm(frames):
    processed_frames = np.array([preprocess_frame(frame) for frame in frames])
    processed_frames = np.expand_dims(processed_frames, axis=0)
    predictions = cnn_lstm_model.predict(processed_frames)
    return predictions  # Adjust return as per your model's output

# Function to detect action
def detect_action(results, prev_results):
    if results.pose_landmarks is None:
        return 'No Human Detected'

    landmarks = results.pose_landmarks.landmark
    if prev_results is None or prev_results.pose_landmarks is None:
        return 'Unknown'

    prev_landmarks = prev_results.pose_landmarks.landmark

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

# Main function to process video file
def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return "Error: Couldn't open video file."

    frames_buffer = []  # Buffer to hold frames
    frames_sequence_length = 20  # Increased sequence length for LSTM

    # Additional variables for fall detection
    prev_hip_y = None
    fall_threshold = 0.2  # Increased threshold for hip Y-coordinate change
    fallen = False
    fall_confirmations = 0
    fall_threshold_confirmations = 5  # Increased confirmations required for fall detection
    no_fall_detected = True  # Flag to check if any fall was detected
    human_detected = False  # Flag to check if any human was detected

    # Prepare to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    prev_results = None
    detected_actions = []
    processed_actions = []
    action_counts = {}
    action_sequence = []  # Store the sequence of actions
    fall_incidents = []  # Store fall incidents

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks is not None:
            human_detected = True
            frame, fallen, fall_confirmations = display_gait(frame, results, prev_results, fallen, fall_confirmations, fall_threshold_confirmations)
            
            frames_buffer.append(frame)
            if len(frames_buffer) == frames_sequence_length:
                predictions = detect_gait_with_cnn_lstm(frames_buffer)
                if np.mean(predictions) > 0.5:
                    fall_confirmations += 1
                    if fall_confirmations >= fall_threshold_confirmations:
                        fallen = True
                        no_fall_detected = False  # Fall detected
                        cv2.putText(frame, "FALL DETECTED ", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        fall_incidents.append((frame, cap.get(cv2.CAP_PROP_POS_MSEC)))
                else:
                    fall_confirmations = 0  # Reset confirmations if fall is not detected
                    
                frames_buffer.pop(0)

            action_label = detect_action(results, prev_results) if prev_results else 'Unknown'
            prev_results = results

        else:
            action_label = 'No Human Detected'

        detected_actions.append(action_label)

        if action_label not in action_counts:
            action_counts[action_label] = 1
        else:
            action_counts[action_label] += 1

        if action_counts[action_label] > 3 and action_label not in processed_actions:
            processed_actions.append(action_label)
            action_counts = {action_label: 4}  # Reset action counts, keep the current action

        action_sequence.append((action_label, cap.get(cv2.CAP_PROP_POS_MSEC)))

        out.write(frame)
        
    cap.release()
    out.release()

    if 'No Human Detected' in processed_actions:
        return "No Human Detected", "", "", ["No Human Detected"], []

    fall_analysis = analyze_action_sequence(action_sequence)
    fall_risk_analysis = detailed_fall_risk_analysis(action_sequence)
    results_string = f"Fall Detection: {'Fall detected' if not no_fall_detected else 'No fall detected'}\n\n\n ACTION SEQUENCE ANALYSIS : {fall_analysis}\n\n\n FALL RISK ANALYSIS : {fall_risk_analysis}\n\n\nActions Detection: " + "\n".join(processed_actions)

    # Save fall incidents as images and aggregate within 5-second intervals
    fall_incident_images = []
    aggregated_fall_incidents = []
    current_interval = []
    for i, (frame, timestamp) in enumerate(fall_incidents):
        if current_interval and timestamp - current_interval[0][1] > 5000:  # 5 seconds interval
            aggregated_fall_incidents.append(current_interval)
            current_interval = []
        current_interval.append((frame, timestamp))
        image_path = os.path.join('static', f'fall_{timestamp}.jpg')
        cv2.imwrite(image_path, frame)
        fall_incident_images.append((image_path, timestamp))
    if current_interval:
        aggregated_fall_incidents.append(current_interval)

    return results_string, fall_analysis, fall_risk_analysis, processed_actions, aggregated_fall_incidents

def analyze_action_sequence(action_sequence):
    action_counts = {
        'Walking': 0,
        'Standing': 0,
        'Sitting': 0,
        'Laying Down': 0,
        'Sitting Down': 0,
        'Standing Up': 0
    }
    action_sequence_length = len(action_sequence)
    fall_detected = False  # Track if a fall was detected
    cause_of_fall = "The cause of the fall is unclear."

    for action, timestamp in action_sequence:
        if action in action_counts:
            action_counts[action] += 1
        if action == 'Laying Down':
            fall_detected = True  # A fall is detected if laying down is present

    if fall_detected:
        if action_sequence_length > 1:
            last_action = action_sequence[-2][0]  # Get the action just before "Laying Down"
            if last_action == 'Walking':
                cause_of_fall = "The person may have fallen due to loss of balance while walking."
            elif last_action == 'Standing':
                cause_of_fall = "The person may have fallen while transitioning from standing."
            elif last_action == 'Sitting':
                cause_of_fall = "The person may have fallen while attempting to stand up."
            elif last_action == 'Sitting Down':
                cause_of_fall = "The person may have lost balance while sitting down."
            # Additional conditions can be added for other actions

        # Calculate percentages for all actions
        walking_percentage = (action_counts['Walking'] / action_sequence_length) * 100 if action_sequence_length > 0 else 0
        standing_percentage = (action_counts['Standing'] / action_sequence_length) * 100 if action_sequence_length > 0 else 0
        sitting_percentage = (action_counts['Sitting'] / action_sequence_length) * 100 if action_sequence_length > 0 else 0
        laying_down_percentage = (action_counts['Laying Down'] / action_sequence_length) * 100 if action_sequence_length > 0 else 0

        action_analysis = []
        if walking_percentage > 0:
            action_analysis.append(f"walking ({walking_percentage:.2f}%)")
        if standing_percentage > 0:
            action_analysis.append(f"standing ({standing_percentage:.2f}%)")
        if sitting_percentage > 0:
            action_analysis.append(f"sitting ({sitting_percentage:.2f}%)")
        if laying_down_percentage > 0:
            action_analysis.append(f"laying down ({laying_down_percentage:.2f}%)")

        if action_analysis:
            return f"The person was " + ', '.join(action_analysis) + " for an extended period of time. "
        else:
            return "The person's actions are unclear. "

    walking_percentage = (action_counts['Walking'] / action_sequence_length) * 100 if action_sequence_length > 0 else 0
    standing_percentage = (action_counts['Standing'] / action_sequence_length) * 100 if action_sequence_length > 0 else 0
    sitting_percentage = (action_counts['Sitting'] / action_sequence_length) * 100 if action_sequence_length > 0 else 0
    laying_down_percentage = (action_counts['Laying Down'] / action_sequence_length) * 100 if action_sequence_length > 0 else 0

    action_analysis = []
    if walking_percentage > 0:
        action_analysis.append(f"walking ({walking_percentage:.2f}%)")
    if standing_percentage > 0:
        action_analysis.append(f"standing ({standing_percentage:.2f}%)")
    if sitting_percentage > 0:
        action_analysis.append(f"sitting ({sitting_percentage:.2f}%)")
    if laying_down_percentage > 0:
        action_analysis.append(f"laying down ({laying_down_percentage:.2f}%)")

    if action_analysis:
        return f"The person was " + ', '.join(action_analysis) + " for an extended period of time."
    else:
        return "The person's actions are unclear."

def detailed_fall_risk_analysis(action_sequence):
    fall_risk_factors = []
    fall_risk_factors.append("Unstable walking pattern") if any(action == 'Walking' for action, _ in action_sequence) else None
    fall_risk_factors.append("Sudden changes in posture") if any(action == 'Sitting Down' or action == 'Standing Up ' for action, _ in action_sequence) else None
    fall_risk_factors.append("Prolonged sitting or lying down") if any(action == 'Sitting' or action == 'Laying Down' for action, _ in action_sequence) else None
    fall_risk_factors.append("Near-fall incidents") if any(action == 'Sitting Down' or action == 'Standing Up ' for action, _ in action_sequence) else None

    if fall_risk_factors:
        return f"Potential fall risk factors detected: {', '.join(fall_risk_factors)}"
    else:
        return "No potential fall risk factors detected."

def detect_laying_down(results):
    landmarks = results.pose_landmarks.landmark

    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    shoulder_hip_diff = abs(left_shoulder.y - left_hip.y) + abs(right_shoulder.y - right_hip.y)

    if shoulder_hip_diff < 0.05:
        return True
    return False

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

    if prev_hip_knee_ratio > 1 and hip_knee_ratio < 1:
        return True
    return False

def detect_sitting(results):
    landmarks = results.pose_landmarks.landmark

    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    hip_knee_ratio = (left_hip.y + right_hip.y) / (left_knee.y + right_knee.y)

    if 0.95 <= hip_knee_ratio <= 1.05 and left_hip.y > 0.5 and right_hip.y > 0.5:
        return True
    return False

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

    if prev_hip_knee_ratio < 1 and hip_knee_ratio > 1:
        return True
    return False

def detect_standing(results):
    landmarks = results.pose_landmarks.landmark

    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    if left_hip.y < 0.5 and right_hip.y < 0.5 and left_knee.y > 0.5 and right_knee.y > 0.5:
        return True
    return False

def detect_walking(results, prev_results):
    landmarks = results.pose_landmarks.landmark
    prev_landmarks = prev_results.pose_landmarks.landmark

    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    prev_left_ankle = prev_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    prev_right_ankle = prev_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    ankle_movement = (left_ankle.x - prev_left_ankle.x) * (right_ankle.x - prev_right_ankle.x)

    if ankle_movement < 0:
        return True
    return False

def display_gait(frame, results, prev_results, fallen, fall_confirmations, fall_threshold_confirmations):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        
        if left_shoulder and left_hip and left_knee:
            angle = calculate_angle(left_shoulder, left_hip, left_knee)
            cv2.putText(frame, f"Angle: {angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

        if left_hip and right_hip:
            hip_y = (left_hip.y + right_hip.y) / 2  # Average hip Y-coordinate

            if prev_results is not None and prev_results.pose_landmarks is not None:
                prev_hip_y = (prev_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y + prev_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
                hip_acceleration = (hip_y - prev_hip_y) / (1 / 30)  # Assuming 30 FPS
                if hip_acceleration > 0.2:
                    fall_confirmations += 1
                    if fall_confirmations >= fall_threshold_confirmations:
                        fallen = True
                        cv2.putText(frame, "FALL!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    fall_confirmations = 0  # Reset confirmations if fall is not detected
            prev_results = results

    return frame, fallen, fall_confirmations


def format_feedback_for_html(feedback_text):
    """
    Convert feedback text into HTML format as a paragraph.
    This function removes metadata, handles bold text formatting, and ensures proper HTML formatting without redundant tags.
    
    Args:
        feedback_text (str): The feedback text to format.

    Returns:
        str: Formatted feedback in HTML for display in the template.
    """
    # Remove metadata, specifically anything following 'additional_kwargs='
    feedback_text = re.split(r'additional_kwargs=', feedback_text)[0].strip()
    
    # Replace asterisks used for bold formatting with <strong> tags, without wrapping empty content
    feedback_text = re.sub(r'\*(.+?)\*', r'<strong>\1</strong>', feedback_text)
    
    # Remove 'content=' prefix
    feedback_text = feedback_text.replace('content=', '').strip()
    
    # Return the formatted content wrapped in a <p> tag
    return f'<p>{feedback_text}</p>'




def generate_video_feedback(fall_detection, action_sequence_analysis, fall_risk_analysis, processed_actions, fall_incidents):
    insight_prompt = insight_prompt_template.invoke({
        "fall_detection": fall_detection,
        "action_sequence_analysis": action_sequence_analysis,
        "fall_risk_analysis": fall_risk_analysis,
        "processed_actions": ', '.join(processed_actions),
        "fall_incidents": fall_incidents
    })

    recommendation_prompt = recommendation_prompt_template.invoke({
        "fall_detection": fall_detection,
        "action_sequence_analysis": action_sequence_analysis,
        "fall_risk_analysis": fall_risk_analysis,
        "processed_actions": ', '.join(processed_actions),
        "fall_incidents": fall_incidents
    })

    summary_prompt = summary_prompt_template.invoke({
        "fall_detection": fall_detection,
        "action_sequence_analysis": action_sequence_analysis,
        "fall_risk_analysis": fall_risk_analysis,
        "processed_actions": ', '.join(processed_actions),
        "fall_incidents": fall_incidents
    })

    def retry_with_exponential_backoff(prompt, max_retries=5):
        retries = 0
        while retries < max_retries:
            try:
                response = llm.invoke(prompt)
                return output_parser.parse(response)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get('retry-after', 1))
                    print(f"Rate limit exceeded. Retrying in {retry_after} seconds...")
                    time.sleep(retry_after)
                    retries += 1
                else:
                    raise e
        raise Exception("Max retries exceeded")

    try:
        insight_response = retry_with_exponential_backoff(insight_prompt)
        insight_content = str(insight_response).strip()
        formatted_insight = format_feedback_for_html(insight_content)

        recommendation_response = retry_with_exponential_backoff(recommendation_prompt)
        recommendation_content = str(recommendation_response).strip()
        formatted_recommendation = format_feedback_for_html(recommendation_content)

        summary_response = retry_with_exponential_backoff(summary_prompt)
        summary_content = str(summary_response).strip()
        formatted_summary = format_feedback_for_html(summary_content)

        return formatted_insight, formatted_recommendation, formatted_summary
    except Exception as e:
        print(f"An error occurred while generating video feedback: {e}")
        return "Error in generating insights.", "Error in generating recommendations.", "Error in generating summary."
    

    
# For Flask application
@app.route('/')
def index12():
    return render_template('index12.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return redirect(url_for('index12'))

    file = request.files['video']
    if file.filename == '':
        return redirect(url_for('index12'))
    
    # Check if the file is an MP4 video
    if not file.filename.endswith('.mp4'):
        return "Error: Please upload a valid MP4 video file.", 400

    upload_folder = 'uploads'
    output_folder = 'output'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_path = os.path.join(upload_folder, file.filename)
    output_path = os.path.join(output_folder, 'output_' + file.filename)
    
    file.save(file_path)
    
    result, fall_analysis, fall_risk_analysis, processed_actions, aggregated_fall_incidents = process_video(file_path, output_path)
    
    # Generate detailed feedback using Groq API
    insight, recommendation, summary = generate_video_feedback(
        fall_detection="Fall detected" if "Fall detected" in result else "No fall detected",
        action_sequence_analysis=fall_analysis,
        fall_risk_analysis=fall_risk_analysis,
        processed_actions=processed_actions,
        fall_incidents=aggregated_fall_incidents
    )
    
    # Format feedback for HTML display
    insight_html = format_feedback_for_html(insight)
    recommendation_html = format_feedback_for_html(recommendation)
    summary_html = format_feedback_for_html(summary)

    # Convert aggregated_fall_incidents to a format that can be easily rendered in the template
    aggregated_fall_incidents_for_template = []
    for interval in aggregated_fall_incidents:
        interval_for_template = []
        for frame, timestamp in interval:
            image_path = f'fall_{timestamp}.jpg'
            interval_for_template.append((image_path, timestamp))
        aggregated_fall_incidents_for_template.append(interval_for_template)
    
    # Render the results template with the formatted HTML content
    return render_template(
        'results.html',
        result=result,
        video_path=output_path,
        fall_analysis=fall_analysis,
        fall_risk_analysis=fall_risk_analysis,
        processed_actions=processed_actions,
        fall_incidents=aggregated_fall_incidents_for_template,
        insight=insight_html,
        recommendation=recommendation_html,
        summary=summary_html
    )

@app.route('/exit')
def exit_system():
    return "System Exited", 200

if __name__ == "__main__":
    app.run()
    #19/11 
    #TEST
    #SETTLED UPLOAD VIDEO
 
