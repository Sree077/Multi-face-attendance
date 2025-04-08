import face_recognition
import cv2
import os
import pickle
from pathlib import Path


def generate_encodings():
    # Directory containing student photos
    PHOTOS_DIR = "dataset"
    ENCODINGS_FILE = "encodings.pkl"

    print("Starting face encoding generation...")

    # Check if dataset directory exists
    if not os.path.exists(PHOTOS_DIR):
        print(f"Error: {PHOTOS_DIR} directory not found!")
        print("Please ensure the dataset folder containing student photos is present.")
        return

    # Initialize lists to store encodings and names
    known_encodings = []
    known_names = []

    # Supported image formats
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

    # Process each image in the directory and its subdirectories
    for image_path in Path(PHOTOS_DIR).rglob('*'):
        if image_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        # Get student name from parent directory name
        student_name = image_path.parent.name
        print(f"Processing {student_name}'s photo: {image_path.name}...")

        try:
            # Load image
            image = face_recognition.load_image_file(str(image_path))

            # Find face locations
            face_locations = face_recognition.face_locations(image, model="hog")

            if not face_locations:
                print(f"No face found in {image_path.name}. Skipping...")
                continue

            # Generate face encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)

            if face_encodings:
                for encoding in face_encodings:
                    known_encodings.append(encoding)
                    known_names.append(student_name)
                print(f"Successfully encoded {len(face_encodings)} face(s) from {image_path.name}")
            else:
                print(f"Could not generate encoding for {image_path.name}")

        except Exception as e:
            print(f"Error processing {image_path.name}: {str(e)}")

    if not known_encodings:
        print("No face encodings were generated. Please check the photos and try again.")
        return

    # Save the encodings to a file
    try:
        data = {
            "encodings": known_encodings,
            "names": known_names
        }
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(data, f)
        print(f"\nSuccessfully generated encodings for {len(known_encodings)} faces.")
        unique_names = set(known_names)
        print(f"Unique students: {len(unique_names)}")
        for name in unique_names:
            print(f"- {name}")
        print(f"\nEncodings saved to {ENCODINGS_FILE}")

    except Exception as e:
        print(f"Error saving encodings: {str(e)}")


if __name__ == "__main__":
    generate_encodings()
