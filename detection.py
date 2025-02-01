import torch
import os
from pathlib import Path
import pandas as pd
import cv2
import time

# Paths to models and directories
custom_model_path = r'C:\Users\HP\Desktop\MINOR\experiment_5122\weights\best.pt'
yolov5_repo_path = r'C:\Users\HP\Desktop\MINOR\yolov5'  # Local path to YOLOv5 repo
frames_dir = r'C:\Users\HP\Desktop\MINOR\all_frames'
output_dir = r'C:\Users\HP\Desktop\MINOR\bounding_box'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load models from local YOLOv5 repository
custom_model = torch.hub.load(yolov5_repo_path, 'custom', path=custom_model_path, source='local', force_reload=True)
pretrained_model = torch.hub.load(yolov5_repo_path, 'yolov5s', source='local', pretrained=True)

# Non-Maximum Suppression function
def apply_nms(detections, iou_threshold=0.5):
    if len(detections) == 0:
        return detections

    boxes = [(int(row['xmin']), int(row['ymin']), int(row['xmax'] - row['xmin']), int(row['ymax'] - row['ymin'])) 
             for _, row in detections.iterrows()]
    scores = detections['confidence'].tolist()

    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.4, nms_threshold=iou_threshold)
    if indices is None or len(indices) == 0:
        return pd.DataFrame(columns=detections.columns)

    indices = [int(i) for i in indices.flatten()] if len(indices.shape) > 1 else [int(i) for i in indices]
    detections = detections.iloc[indices]
    return detections

# Function to process detections and save output
def process_detections(image, image_file_name=None):
    # 1. Inference with both YOLO models
    pretrained_results = pretrained_model(image)
    general_detections = pretrained_results.pandas().xyxy[0]
    
    custom_results = custom_model(image)
    medicine_detections = custom_results.pandas().xyxy[0]
    
    # 2. Combine detections and apply NMS
    combined_detections = pd.concat([general_detections, medicine_detections])
    combined_detections = apply_nms(combined_detections)

    # 3. Draw bounding boxes on the image
    for idx, row in combined_detections.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row.get('name', 'unknown')
        confidence = row.get('confidence', 0.0)
        color = (0, 255, 0) if label != 'medicine' else (0, 0, 255)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 4. Save or display image with detections
    if image_file_name:
        output_image_path = Path(output_dir) / image_file_name
        cv2.imwrite(str(output_image_path), image)
        print(f"Saved combined detection image to {output_image_path}")
    else:
        cv2.imshow("Detections", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit camera detection
            return False
    return True

# Main function to select mode and run detection
def main():
    mode = input("Enter 'camera' to use live camera feed or 'frames' to process saved frames: ").strip().lower()

    if mode == 'camera':
        # Open camera feed for detection
        cap = cv2.VideoCapture(0)  # Try index 0 first
        time.sleep(2)  # Wait for the camera to initialize
        
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)  # Try index 1 if index 0 fails
            if not cap.isOpened():
                print("Error: Could not open camera.")
                return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera.")
                break
            
            # Process detections on each frame from the camera
            if not process_detections(frame):
                break  # Exit loop if 'q' is pressed

        cap.release()
        cv2.destroyAllWindows()

    elif mode == 'frames':
        # Process images from frames directory
        for image_file in os.listdir(frames_dir):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = Path(frames_dir) / image_file
                image = cv2.imread(str(image_path))
                
                if image is not None:
                    process_detections(image, image_file_name=image_file)
                else:
                    print(f"Warning: Could not load image {image_file}")

    else:
        print("Invalid input. Please enter 'camera' or 'frames'.")

if __name__ == "__main__":
    main()
