import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
import os

# --- Configuration ---
# NOTE: Update the MODEL_PATH to the actual location of your model file
MODEL_PATH = r"C:\Users\Thilak D G\OneDrive\Desktop\dataset PYthon\cnn_age_gender_model.h5"

# NOTE: Update the paths list to include the correct locations for your test images
paths = [
    r"C:\Users\Thilak D G\OneDrive\Desktop\dataset PYthon\img1.png",
    r"C:\Users\Thilak D G\OneDrive\Desktop\dataset PYthon\img2.png",
    r"C:\Users\Thilak D G\OneDrive\Desktop\dataset PYthon\img3.png",
    r"C:\Users\Thilak D G\OneDrive\Desktop\dataset PYthon\img4.png",
    r"C:\Users\Thilak D G\OneDrive\Desktop\dataset PYthon\img5.png",
]
# ---------------------

# Load the model. We use compile=False since we are only using it for inference.
try:
    model = load_model(MODEL_PATH, compile=False)
    print(f"✅ Successfully loaded model from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error: Could not load the model file. Please check the path.")
    print(f"Error details: {e}")
    exit()

results = []

print("\nProcessing images and generating predictions...")

for p in paths:
    # 1. Load Image
    img = cv2.imread(p)
    if img is None:
        print(f"❌ Could not load: {os.path.basename(p)}. Skipping.")
        continue

    # 2. Preprocessing (must match training)
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize and Normalize
    img_resized_normalized = cv2.resize(img_rgb, (128, 128)) / 255.0
    # Add batch dimension (1, 128, 128, 3)
    inp = np.expand_dims(img_resized_normalized, axis=0)

    # 3. Predict
    # pred_age is index 0, pred_gender is index 1
    # Both return a shape of (1, 1)
    pred_age, pred_gender = model.predict(inp, verbose=0)

    # 4. Post-processing
    age = float(pred_age[0][0])
    conf_raw = float(pred_gender[0][0]) # This is the probability of Female (Class 1)

    # Determine gender and the true confidence score (always >= 0.5)
    if conf_raw >= 0.5:
        gender = "Female"
        conf = conf_raw  # Confidence in Female is the raw sigmoid output
    else:
        gender = "Male"
        conf = 1.0 - conf_raw  # Confidence in Male is 1 - raw sigmoid output

    results.append((os.path.basename(p), age, gender, conf))

# 5. Output Results
print("\n" + "="*30)
print("🔍 FINAL PREDICTIONS")
print("="*30)

for r in results:
    # r[0]: Image Name, r[1]: Age, r[2]: Gender, r[3]: Confidence
    print(f"Image: {r[0]:<10} Age: {r[1]:>5.2f} | Gender: {r[2]:<6} | Confidence: {r[3]:.2f}")

print("="*30)
print("\n✅ Prediction complete!")