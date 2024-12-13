import os
import cv2
from ultralytics import YOLO

# Model YOLO
face_detector = YOLO('yolov8n-face.pt')

# Folder input dan output
input_folder = 'Dataset/'
output_folder = 'Preprocessing_Dataset/'

os.makedirs(output_folder, exist_ok=True)

for person in os.listdir(input_folder):
    person_folder = os.path.join(input_folder, person)
    output_person_folder = os.path.join(output_folder, person)
    os.makedirs(output_person_folder, exist_ok=True)

    for img_file in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_file)
        image = cv2.imread(img_path)

        results = face_detector(image)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = image[y1:y2, x1:x2]
                
                # Save cropped face
                output_path = os.path.join(output_person_folder, img_file)
                cv2.imwrite(output_path, face)
