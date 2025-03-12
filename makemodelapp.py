import os
import tensorflow as keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# Check GPU availability
print("Num GPUs Available:", len(keras.config.list_physical_devices('GPU')))

# Define absolute dataset path
dataset_path = r"C:\Users\User\Desktop\Python\HandMotionDetectionAI\dataset"

# Check if dataset path exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

# Image preprocessing with augmentation
data_gen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Load training data
train_data = data_gen.flow_from_directory(
    dataset_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

# Load validation data
val_data = data_gen.flow_from_directory(
    dataset_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),  # Extra dropout for regularization
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(train_data.num_classes, activation="softmax")
])

# Compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Callbacks for training
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
checkpoint_path = r"C:\Users\User\Desktop\Python\HandMotionDetectionAI\best_model.h5"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)

# Train model
model.fit(train_data, validation_data=val_data, epochs=10, callbacks=[lr_scheduler, checkpoint])

# Save final model
final_model_path = r"C:\Users\User\Desktop\Python\HandMotionDetectionAI\hand_sign_model.h5"
model.save(final_model_path)

print(f"Model training complete. Best model saved at '{checkpoint_path}', Final model saved at '{final_model_path}'")
