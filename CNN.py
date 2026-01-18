import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_age_train = np.load("y_age_train.npy")
y_age_test = np.load("y_age_test.npy")
y_gender_train = np.load("y_gender_train.npy")
y_gender_test = np.load("y_gender_test.npy")

# Normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0

# ---------------------------------------------------------
# BUILD CNN MODEL
# ---------------------------------------------------------
inputs = layers.Input(shape=(128, 128, 3))

x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(inputs)
x = layers.MaxPooling2D(2,2)(x)

x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D(2,2)(x)

x = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D(2,2)(x)

x = layers.Conv2D(256, (3,3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D(2,2)(x)

x = layers.Flatten()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation="relu")(x)

# Age output (regression)
age_output = layers.Dense(128, activation="relu")(x)
age_output = layers.Dense(1, name="age_output")(age_output)

# Gender output (classification)
gender_output = layers.Dense(128, activation="relu")(x)
gender_output = layers.Dense(1, activation="sigmoid", name="gender_output")(gender_output)

model = models.Model(inputs=inputs, outputs=[age_output, gender_output])
model.summary()

# ---------------------------------------------------------
# COMPILE MODEL
# ---------------------------------------------------------
model.compile(
    optimizer=optimizers.Adam(0.0008),
    loss={
        "age_output": "mae",
        "gender_output": "binary_crossentropy"
    },
    metrics={
        "age_output": "mae",
        "gender_output": "accuracy"
    }
)

# ---------------------------------------------------------
# CALLBACKS (with fix)
# ---------------------------------------------------------
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_gender_output_accuracy",
        factor=0.5,
        patience=3,
        mode="max"
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_gender_output_accuracy",
        patience=6,
        mode="max",
        restore_best_weights=True
    )
]

# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------
history = model.fit(
    X_train,
    {"age_output": y_age_train, "gender_output": y_gender_train},
    validation_data=(X_test, {"age_output": y_age_test, "gender_output": y_gender_test}),
    epochs=35,
    batch_size=64,
    callbacks=callbacks
)

model.save("cnn_age_gender_model.h5")
print("\n🎉 Model Saved: cnn_age_gender_model.h5")

# ---------------------------------------------------------
# GRAPH 1 — GENDER ACCURACY
# ---------------------------------------------------------
plt.figure()
plt.plot(history.history["gender_output_accuracy"])
plt.plot(history.history["val_gender_output_accuracy"])
plt.title("Gender Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.savefig("gender_accuracy.png")
plt.close()

# ---------------------------------------------------------
# GRAPH 2 — AGE MAE
# ---------------------------------------------------------
plt.figure()
plt.plot(history.history["age_output_mae"])
plt.plot(history.history["val_age_output_mae"])
plt.title("Age MAE (Lower is Better)")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend(["Train", "Validation"])
plt.savefig("age_mae.png")
plt.close()

# ---------------------------------------------------------
# GRAPH 3 — AGE ACCURACY (Tolerance method)
# ---------------------------------------------------------
print("\n📏 Calculating Age Accuracy...")

pred_age, _ = model.predict(X_test)
pred_age = pred_age.reshape(-1)

TOLERANCE = 5  # ±5 years

correct = np.abs(pred_age - y_age_test) <= TOLERANCE
age_accuracy = np.mean(correct)

print(f"🎯 Age Accuracy (±{TOLERANCE} years): {age_accuracy:.4f}")

plt.figure(figsize=(10,4))
plt.plot(correct, linewidth=0.5)
plt.title(f"Age Accuracy per Sample (±{TOLERANCE} years)")
plt.xlabel("Test Sample Index")
plt.ylabel("Correct(1)/Wrong(0)")
plt.savefig("age_accuracy.png")
plt.close()

print("\n📊 Saved: gender_accuracy.png, age_mae.png, age_accuracy.png")
print("✅ Training Complete!")