import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import google.generativeai as genai
from speak import speak  # Import the speak function
from dotenv import load_dotenv
from os import getenv
import threading  # For running speak function without blocking
import sys
import tkinter as tk
from tkinter import ttk  # For modern UI widgets
from PIL import Image, ImageTk
from ttkthemes import ThemedTk  # For modern themes

# Load environment variables
load_dotenv()

# Set up Gemini API
KEY1 = getenv("KEY1AM")  # Ensure your .env file has KEY1AM=YOUR_GEMINI_API_KEY
if not KEY1:
    print("Please set your Gemini API key in the .env file.")
    sys.exit(1)

genai.configure(api_key=KEY1)
try:
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    sys.exit(1)

conversation_history = []

def get_response(user_message):
    """
    Gets a response from Gemini while maintaining conversation history.
    """
    global conversation_history
    conversation_history.append({"role": "user", "content": user_message})
    prompt = ""
    for turn in conversation_history:
        prompt += f"{turn['role']}: {turn['content']}\n"
    prompt += "assistant: "
    # Instruct Gemini to interpret incomplete sentences
    response = model.generate_content(prompt, max_output_tokens=256, temperature=0.7)
    conversation_history.append({"role": "assistant", "content": response.text})
    return response.text

# Print welcome message
print("Welcome to the Airport Virtual Navigation Assistant!")

# Load the trained gesture recognition model
MODEL_PATH = './model.p'
NUM_LANDMARKS = 21

try:
    with open(MODEL_PATH, 'rb') as f:
        model_dict = pickle.load(f)
    gesture_model = model_dict['model']
except FileNotFoundError:
    print("Model file not found. Ensure 'model.p' exists in the current directory.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Gesture labels (Removed 'english' by excluding label 11)
labels_dict = {
    0: 'I', 1: 'restaurant', 2: 'security', 3: 'soon', 4: 'ticket', 5: 'time',
    6: 'want', 7: 'is', 8: 'bathroom', 9: 'begin', 10: 'delay',
    # 11: 'english',  # Disabled 'english'
    12: 'flight', 13: 'gate', 14: 'help', 15: 'where', 16: 'stop'
}

# Setup MediaPipe Hands
cap = cv2.VideoCapture(1)  # Use 0 for the default camera
if not cap.isOpened():
    print("Camera not initialized properly. Check camera index or permissions.")
    sys.exit(1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

# Initialize variables
gesture_counts = {}
sentence = ""
api_response = ""

begin_count = 0
stop_gate_count = 0
sentence_started = False

def extract_landmarks(hand_landmarks):
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.append(landmark.x)
        landmarks.append(landmark.y)
    return landmarks

def speak_response(text):
    threading.Thread(target=speak, args=(text,)).start()

# Create Themed Tkinter window
root = ThemedTk(theme="equilux")  # Choose a modern theme
root.title("Airport Virtual Navigation Assistant")
root.geometry("900x700")

# Configure styles
style = ttk.Style(root)
style.configure('TLabel', font=('Helvetica', 14))
style.configure('TButton', font=('Helvetica', 12))
style.configure('TFrame', background=style.lookup('TLabel', 'background'))

# Create frames
video_frame = ttk.Frame(root)
video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

info_frame = ttk.Frame(root)
info_frame.pack(side=tk.BOTTOM, fill=tk.X)

# Video display label
video_label = ttk.Label(video_frame)
video_label.pack()

# Info labels
gesture_label = ttk.Label(info_frame, text="Gesture: ")
gesture_label.pack(pady=5)

detected_word_label = ttk.Label(info_frame, text="Detected Word: ")
detected_word_label.pack(pady=5)

sentence_label = ttk.Label(info_frame, text="Sentence: ")
sentence_label.pack(pady=5)

response_label = ttk.Label(info_frame, text="Response: ", wraplength=880, justify="left")
response_label.pack(pady=5)

def update_frame():
    global gesture_counts, sentence, api_response
    global begin_count, stop_gate_count, sentence_started
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        root.after(10, update_frame)
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    left_hand_landmarks = [0.0] * (NUM_LANDMARKS * 2)
    right_hand_landmarks = [0.0] * (NUM_LANDMARKS * 2)

    if results.multi_hand_landmarks:
        for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            if handedness.classification[0].label == "Left":
                left_hand_landmarks = extract_landmarks(hand_landmarks)
            elif handedness.classification[0].label == "Right":
                right_hand_landmarks = extract_landmarks(hand_landmarks)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    try:
        features = np.concatenate([left_hand_landmarks, right_hand_landmarks])
        prediction = gesture_model.predict([features])[0]
        predicted_character = labels_dict.get(prediction, "Unknown")
    except Exception as e:
        print(f"Error during prediction: {e}")
        predicted_character = "Error"

    # Update the gesture label
    gesture_label.config(text=f"Gesture: {predicted_character}")

    # Update the detected word label
    detected_word_label.config(text=f"Detected Word: {predicted_character}")

    # Handle the logic for "begin" and "stop"/"gate" gestures
    if predicted_character == "begin":
        begin_count += 1
        if begin_count >= 3 and not sentence_started:
            sentence_started = True
            begin_count = 0  # Reset count
            gesture_counts = {}  # Reset gesture counts
            sentence = ""
            sentence_label.config(text="Sentence: ")
            response_label.config(text="Response: ")
            print("Sentence formation started.")
    else:
        begin_count = 0  # Reset if other gesture detected

    if predicted_character in ["stop", "gate"]:
        stop_gate_count += 1
        if stop_gate_count >= 3 and sentence_started:
            sentence_started = False
            stop_gate_count = 0  # Reset count
            # Send sentence to Gemini
            if sentence.strip() != "":
                # Instruct Gemini to interpret incomplete sentences
                full_message = sentence.strip() + " (You are an assistant at Chennai airport. Interpret the user's intent even if the sentence is incomplete or contains only a few words. Provide concise, formal, and informative responses as an airport assistant would.)"
                assistant_response = get_response(full_message)
                response_label.config(text=f"Response: {assistant_response}")
                speak_response(assistant_response)
                sentence = ""
                sentence_label.config(text="Sentence: ")
                print("Assistant response:", assistant_response)
            gesture_counts = {}  # Reset gesture counts
    else:
        stop_gate_count = 0  # Reset if other gesture detected

    if sentence_started and predicted_character not in ["Unknown", "Error", "begin", "stop", "gate"]:
        # Update the count for the predicted gesture
        gesture_counts[predicted_character] = gesture_counts.get(predicted_character, 0) + 1

        # If a gesture is recognized more than 5 times, add it to the sentence and reset its count
        if gesture_counts[predicted_character] >= 5:
            sentence += predicted_character + " "
            sentence_label.config(text=f"Sentence: {sentence}")
            gesture_counts[predicted_character] = 0  # Reset count
            print(f"Added '{predicted_character}' to sentence. Current sentence: '{sentence}'")

    # Convert frame to image for Tkinter
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Schedule the next frame update
    root.after(10, update_frame)

def on_closing():
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    root.destroy()
    sys.exit()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Start the video loop
update_frame()

# Start the Tkinter main loop
root.mainloop()
