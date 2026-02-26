import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
import pyttsx3
import threading

# --- 1. SETUP YOUR CUSTOM AI BRAIN ---
print("Loading Custom Brain...")
try:
    model = pickle.load(open('custom_model.p', 'rb'))
except FileNotFoundError:
    print("Error: custom_model.p not found. Did you run the training script?")
    exit()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)

cap = cv2.VideoCapture(0)

current_sentence = ""
last_char = ""
hold_time = 0
CONFIDENCE_THRESHOLD = 0.80  # Must be 80% sure before typing

# --- 2. SETUP AUDIO ENGINE (Threading) ---
def speak_text(text):
    if not text.strip(): return 
    def run_speech():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150) # Speaking speed
        engine.say(text)
        engine.runAndWait()
    
    # Run speech in background so the camera doesn't freeze
    threading.Thread(target=run_speech, daemon=True).start()

print("--- APP STARTED ---")
print("Controls: 'q' to Quit | 'c' to Clear")

# --- 3. MAIN CAMERA LOOP ---
while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            data_aux = []
            base_x = hand_lms.landmark[0].x
            base_y = hand_lms.landmark[0].y
            
            for lm in hand_lms.landmark:
                data_aux.extend([lm.x - base_x, lm.y - base_y])
            
            # Predict using your custom data
            probabilities = model.predict_proba([np.asarray(data_aux)])[0]
            max_confidence = np.max(probabilities)
            prediction = model.classes_[np.argmax(probabilities)]
            
            # Typing & Audio Logic
            if max_confidence >= CONFIDENCE_THRESHOLD:
                # Display letter in GREEN
                cv2.putText(frame, f"{prediction} ({max_confidence*100:.0f}%)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

                if prediction == last_char:
                    if time.time() - hold_time > 1.5:
                        
                        # --- YOUR CUSTOM COMMANDS ---
                        if prediction == "SPACE": 
                            current_sentence += " "
                            
                        elif prediction == "DELETE": 
                            current_sentence = current_sentence[:-1]
                            
                        elif prediction == "AUDIO":
                            speak_text(current_sentence) # Speaks the whole sentence!
                            
                        # Handle normal letters (A, B, C...)
                        else: 
                            current_sentence += prediction
                        
                        hold_time = time.time()
                        cv2.rectangle(frame, (0,0), (w, h), (0, 255, 0), 10) # Flash green 
                else:
                    last_char = prediction
                    hold_time = time.time()
            else:
                # Display letter in RED (Not confident enough to type)
                cv2.putText(frame, f"{prediction} ({max_confidence*100:.0f}%)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                hold_time = time.time() 

    # --- UI DASHBOARD ---
    cv2.rectangle(frame, (0, h-60), (w, h), (0,0,0), -1)
    cv2.putText(frame, f"Sentence: {current_sentence}", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    cv2.imshow('Authentic Sign Language AI', frame)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'): break
    if key & 0xFF == ord('c'): current_sentence = "" 

cap.release()
cv2.destroyAllWindows()