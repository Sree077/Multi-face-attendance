import cv2
import os

video_path = 'Dany.mp4'
person_name = 'Dany Koshy P'  # You can change this as needed
output_folder = os.path.join('dataset', person_name)

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Rotate frame if needed
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Save frame
    cv2.imwrite(f"{output_folder}/frame_{frame_count:04d}.jpg", frame)
    frame_count += 1

cap.release()
