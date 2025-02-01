import cv2
import os

# Function to extract frames from the video
def extract_frames(video_path, output_folder, frame_interval=30):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    frame_count = 0
    extracted_count = 0

    # Loop through the video frames
    while True:
        ret, frame = cap.read()

        # If video has ended, break the loop
        if not ret:
            break

        # Save the frame at the specified interval
        if frame_count % frame_interval == 0:
            # Generate filename for the frame
            frame_filename = os.path.join(output_folder, f"frame_{extracted_count}.jpg")
            # Save the frame as an image
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame {extracted_count} at {frame_filename}")
            extracted_count += 1

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Extraction completed. Total frames saved: {extracted_count}")

# Video file path and output directory
video_file = r"C:\Users\HP\Desktop\MINOR\vid.mp4"  # Path to your video file
output_dir = "all_frames"  # Directory where frames will be saved

frame_interval = 30  # Save every 30th frame (adjust as needed)

# Call the function to extract frames
extract_frames(video_file, output_dir, frame_interval)
