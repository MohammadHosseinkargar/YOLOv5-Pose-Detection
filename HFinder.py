import cv2
import torch
import pandas as pd
import os
from datetime import datetime
import openai
import mediapipe as mp
import time
import numpy as np

# Set your OpenAI API key
openai.api_key = 'Set-your-OpenAI-API-key'

# Load YOLOv5 medium model for better accuracy
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

# Set up MediaPipe Holistic model for 3D pose estimation
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Set up video capture (0 for webcam, replace with file path for video)
cap = cv2.VideoCapture(0)

# Increase resolution for better accuracy
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Create a directory to save detected frames
frames_dir = 'detected_frames'
os.makedirs(frames_dir, exist_ok=True)

# Set up a list to log detection results
log_data = []
log_file = 'detection_log.csv'
confidence_threshold = 0.5  # Confidence threshold for detections
frame_count = 0

# Variables for controlling GPT API calls
last_gpt_call = time.time()
gpt_call_interval = 60  # Limit GPT calls to once every 60 seconds
cached_responses = {}
actions_sequence = []

def get_gpt_response(prompt, max_retries=5, wait_time=5):
    """Function to get a response from GPT-3 with retry on rate limit error."""
    retries = 0
    while retries < max_retries:
        try:
            # Check if the prompt exists in cache
            if prompt in cached_responses:
                return cached_responses[prompt]

            # Make API call
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            gpt_text = response['choices'][0]['message']['content'].strip()

            # Cache the response to avoid repeating the same API call
            cached_responses[prompt] = gpt_text
            return gpt_text
        
        except openai.error.RateLimitError:
            retries += 1
            print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            wait_time *= 2  # Exponential backoff

        except openai.error.OpenAIError as e:
            print(f"OpenAI API error: {e}")
            return "Error retrieving response from GPT."

def estimate_pose_angles(pose_landmarks):
    """Estimate angles between body parts for better pose description."""
    # Example: calculate the angle of the left arm (between shoulder, elbow, wrist)
    def calculate_angle(a, b, c):
        a = np.array(a)  # First point
        b = np.array(b)  # Mid point
        c = np.array(c)  # End point
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    left_arm_angle = calculate_angle(
        [pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x,
         pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y],
        [pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].x,
         pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].y],
        [pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].x,
         pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y])

    return left_arm_angle

def recognize_action(pose_landmarks):
    """Basic rule-based action recognition based on pose."""
    left_arm_angle = estimate_pose_angles(pose_landmarks)
    
    if left_arm_angle > 160:
        return "standing"
    elif left_arm_angle < 30:
        return "raising left arm"
    else:
        return "undefined action"

# Video capture loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection using YOLOv5
    results = model(frame)

    # Convert YOLOv5 results to pandas DataFrame
    detections = results.pandas().xyxy[0] 

    # Pose detection using MediaPipe Holistic (for 3D pose estimation)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    holistic_results = holistic.process(image_rgb)

    # Draw pose landmarks if any are detected
    if holistic_results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, holistic_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
        )

        # Recognize action and add to the sequence
        action = recognize_action(holistic_results.pose_landmarks)
        actions_sequence.append(action)

    # Process detected objects and log those that are 'person'
    detected_parts = []
    for index, row in detections.iterrows():
        if row['name'] == 'person' and row['confidence'] >= confidence_threshold:
            # Bounding box coordinates
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            confidence = row['confidence']

            # Draw rectangle and add text
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f'Person {confidence:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save detected frame
            frame_path = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"Frame {frame_count} saved at {frame_path}")

            # Log detection information
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_data.append([timestamp, 'person', confidence, frame_count])
            detected_parts.append("person")

    # If a person is detected and enough time has passed, get GPT response
    if detected_parts and (time.time() - last_gpt_call) > gpt_call_interval:
        detected_parts_str = ', '.join(detected_parts)
        # Provide the sequence of detected actions to GPT for generating a narrative
        actions_str = ', '.join(actions_sequence[-5:])  # Last 5 actions
        gpt_prompt = f"A person was detected with the following parts visible: {detected_parts_str}. The person has performed these actions: {actions_str}. Describe the scene."
        gpt_response = get_gpt_response(gpt_prompt)
        print("GPT Response:", gpt_response)
        last_gpt_call = time.time()

    # Display the frame
    cv2.imshow('YOLOv5 + 3D Pose + Action Detection + GPT Chat', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Save the log data to a CSV file
pd.DataFrame(log_data, columns=["Timestamp", "Object", "Confidence", "Frame"]).to_csv(log_file, index=False)
print(f"Detection log saved to {log_file}")
