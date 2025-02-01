import os
import cv2
import torch  # For YOLOv5
import pytesseract
from gtts import gTTS

# Specify the path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load your custom YOLO model for medicine detection
medicine_model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\HP\Desktop\MINOR\yolov5\runs\train\exp2\weights\best.pt')

# Set the path to the main folder
main_folder = r"C:\Users\HP\Desktop\MINOR"
audio_output_folder = os.path.join(main_folder, 'audio_outputs')

# Create output folder for audio files
os.makedirs(audio_output_folder, exist_ok=True)

# Function to enhance and clean images for OCR
def preprocess_image(cropped_img):
    """Enhances the cropped image for OCR."""
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    # Enhance contrast
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

# Function to extract and clean medicine details
def extract_medicine_text(text):
    """Cleans the raw OCR text."""
    return ' '.join(text.strip().splitlines()).strip()

# Initialize webcam feed
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or specify another camera index

print("Press 'q' to quit the live detection.")

while True:
    # Capture frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Run the custom medicine detection model
    medicine_results = medicine_model(frame)

    # Extract bounding boxes for detected medicines
    medicine_boxes = medicine_results.xyxy[0]  # Get detections (x1, y1, x2, y2, confidence, class)

    # Process each detection
    for box in medicine_boxes:
        x1, y1, x2, y2, conf, cls = box.tolist()
        if int(cls) == 0:  # Assuming class 0 is for medicines
            # Draw bounding box on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Crop the image to the medicine bounding box
            crop = frame[int(y1):int(y2), int(x1):int(x2)]

            if crop.size == 0:
                print("Empty bounding box detected.")
                continue

            # Preprocess the cropped image
            preprocessed_crop = preprocess_image(crop)

            # Run OCR on the cropped image
            custom_config = r'--oem 3 --psm 6'
            extracted_text = pytesseract.image_to_string(preprocessed_crop, config=custom_config)

            # Extract and clean the text
            audio_text = extract_medicine_text(extracted_text)

            # Display extracted text on the frame
            if audio_text:
                cv2.putText(frame, audio_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Generate audio if readable text is found
                try:
                    tts = gTTS(text=audio_text, lang='en')
                    audio_filename = f"live_text_{int(conf * 100)}.mp3"
                    tts.save(os.path.join(audio_output_folder, audio_filename))
                    print(f"Audio saved as: {audio_filename}")
                except Exception as e:
                    print(f"Failed to generate audio: {e}")
            else:
                print("No readable text found.")

    # Display the frame with detections
    cv2.imshow('Live Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

print("Live detection ended.")
