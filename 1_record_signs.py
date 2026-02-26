import cv2
import mediapipe as mp
import csv
import time
import os

# --- 1. SETUP ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)

FILENAME = "my_authentic_signs.csv"

# Create CSV file with headers if it doesn't exist
if not os.path.exists(FILENAME):
    with open(FILENAME, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["label"]
        for i in range(21):
            header.extend([f"x{i}", f"y{i}"])
        writer.writerow(header)

cap = cv2.VideoCapture(0)

print("--- AUTHENTIC SIGN RECORDER ---")

# --- 2. RECORDING LOOP ---
while True:
    # Ask the user what word they want to sign
    word = input("\nEnter the word you want to sign (e.g., 'Hello', 'Help') or 'q' to quit: ")
    if word.lower() == 'q': 
        break
    
    print(f"\nGet ready to sign '{word}' in 3 seconds...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    
    print(f"🔴 RECORDING '{word}' NOW! Hold the sign and move your hand slightly around...")
    
    samples_collected = 0
    while samples_collected < 100:
        success, frame = cap.read()
        if not success: continue
        
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        # If it sees a hand, save the geometry
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                
                base_x = hand_lms.landmark[0].x
                base_y = hand_lms.landmark[0].y
                
                row = [word.upper()]
                for lm in hand_lms.landmark:
                    row.extend([lm.x - base_x, lm.y - base_y])
                
                # Save to CSV
                with open(FILENAME, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                    
                samples_collected += 1
                
        # Show progress on screen
        cv2.putText(frame, f"Recording '{word.upper()}': {samples_collected}/100", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow("Recorder", frame)
        cv2.waitKey(1)
        
    print(f"✅ Successfully recorded 100 samples for '{word}'!")

cap.release()
cv2.destroyAllWindows()