import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Suppress AVX2 / FMA warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Define paths
data_dir = "dataset/preprocessed_dataset"
train_data_dir = os.path.join(data_dir, "train")
val_data_dir = os.path.join(data_dir, "test")

# Image preprocessing
datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = datagen.flow_from_directory(train_data_dir, target_size=(224, 224), batch_size=32, class_mode="sparse")
val_generator = datagen.flow_from_directory(val_data_dir, target_size=(224, 224), batch_size=32, class_mode="sparse")

# Load MobileNetV2 model as feature extractor
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model initially

# Build the model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
out = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=out)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train the model (initial training phase)
model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=[early_stopping, lr_scheduler])

# Fine-tuning: Unfreeze last few layers and recompile
for layer in base_model.layers[-20:]:  # Unfreezing last 20 layers
    layer.trainable = True

# Compile again with a lower learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model (fine-tuning phase)
history = model.fit(train_generator, validation_data=val_generator, epochs=20, callbacks=[early_stopping, lr_scheduler])

# Convert and save model to .tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("face_recognition_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model training complete, fine-tuned, and saved as face_recognition_model.tflite!")
