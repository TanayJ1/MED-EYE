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
bounding_box_folder = os.path.join(main_folder, 'bounding_box')
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

# Process each image in the bounding_box folder
for filename in os.listdir(bounding_box_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(bounding_box_folder, filename)
        print(f"Processing image: {image_path}")

        # Load the medicine image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Run the custom medicine detection model
        medicine_results = medicine_model(image)

        # Extract bounding boxes for detected medicines
        medicine_boxes = medicine_results.xyxy[0]  # Get detections (x1, y1, x2, y2, confidence, class)

        # Filter and process only medicine detections
        for box in medicine_boxes:
            x1, y1, x2, y2, conf, cls = box.tolist()
            if int(cls) == 0:  # Assuming class 0 is for medicines
                # Crop the image to the medicine bounding box
                crop = image[int(y1):int(y2), int(x1):int(x2)]

                if crop.size == 0:
                    print(f"Empty bounding box detected for {filename}")
                    continue

                # Preprocess the cropped image
                preprocessed_crop = preprocess_image(crop)

                # Run OCR on the cropped image
                custom_config = r'--oem 3 --psm 6'
                extracted_text = pytesseract.image_to_string(preprocessed_crop, config=custom_config)

                # Print the extracted text for debugging
                print("Extracted Text:", extracted_text)

                # Extract and clean the text for audio
                audio_text = extract_medicine_text(extracted_text)

                # Ensure the text is not empty before generating audio
                if audio_text:
                    try:
                        tts = gTTS(text=audio_text, lang='en')
                        audio_filename = f"{os.path.splitext(filename)[0]}_text_{int(conf * 100)}.mp3"
                        tts.save(os.path.join(audio_output_folder, audio_filename))
                        print(f"Audio saved as: {audio_filename}")
                    except Exception as e:
                        print(f"Failed to generate audio for {filename}: {e}")
                else:
                    print(f"No readable text found in image: {filename}")

print("Processing complete. Check the audio_outputs folder for audio files.")
