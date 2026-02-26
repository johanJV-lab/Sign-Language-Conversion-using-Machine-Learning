import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("--- STEP 2: TRAINING CUSTOM AI BRAIN ---")

# 1. Load your custom dataset
print("Loading data from 'my_authentic_signs.csv'...")
try:
    data = pd.read_csv('my_authentic_signs.csv')
except FileNotFoundError:
    print("Error: Could not find the CSV file. Make sure you are in the right folder!")
    exit()

# 2. Separate Features (Coordinates) and Labels (Your Letters/Commands)
X = data.drop('label', axis=1) 
y = data['label']              

# 3. Split Data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

# 4. Train the Random Forest
print("Training the AI on YOUR hands... (This is usually very fast)")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 5. Test the Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Custom Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save the trained model 
with open('custom_model.p', 'wb') as f:
    pickle.dump(model, f)
print("--- DONE! Saved custom AI brain to 'custom_model.p' ---")
