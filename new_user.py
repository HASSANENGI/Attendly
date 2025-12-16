from ultralytics import YOLO
import torch
import cv2
import argparse
import os
import time

def parse_variables():
    parser = argparse.ArgumentParser(description='Run Inference YOLOv8 for Face Detection')
    parser.add_argument('-w', '--weights', type=str, help='Path to trained weights',
                        default=r'Models\yolov8n_100e.pt')
    parser.add_argument('-t', '--threshold', type=float, help='Confidence threshold for saving faces',
                        default=0.5)
    args = parser.parse_args()
    return vars(args)

def main():
    # Parse command-line arguments
    variables = parse_variables()
    weights = variables['weights']
    confidence_threshold = variables['threshold']
    
    # Take input for the folder name where images will be saved
    folder_name = input("Enter the folder name to save cropped faces (inside cropped_faces folder): ").strip()
    
    # Define the base folder for saving images
    base_output_folder = 'cropped_faces'
    output_folder = os.path.join(base_output_folder, folder_name)
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    print(f"Images will be saved in: {output_folder}")
    
    # Load YOLOv8 model
    print(f"Loading YOLOv8 model from {weights}...")
    model = YOLO(weights)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Camera is live. Showing feed with bounding boxes.")
    start_time = time.time()
    frame_count = 0
    last_capture_time = time.time()

    while frame_count < 5:  # Capture exactly 5 images
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Run YOLOv8 detection
        results = model.predict(source=frame, verbose=False)

        # Check and process detected faces
        if results and results[0].boxes.xyxy.shape[0] > 0:  # Ensure results are non-empty
            for box, conf in zip(results[0].boxes.xyxy, results[0].boxes.conf):
                if conf >= confidence_threshold:
                    x1, y1, x2, y2 = map(int, box[:4])
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)

                    # Draw bounding box and confidence score
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Conf: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    # Capture and save cropped face every 5 seconds
                    if time.time() - last_capture_time >= 5:
                        cropped_face = frame[y1:y2, x1:x2]
                        if cropped_face.size > 0:  # Ensure valid crop
                            save_path = os.path.join(output_folder, f'face_{frame_count}.jpg')
                            cv2.imwrite(save_path, cropped_face)
                            print(f"Cropped face saved: {save_path}")
                            frame_count += 1
                            last_capture_time = time.time()
                        break  # Only save one face per frame

        # Display the frame with bounding boxes
        cv2.imshow("YOLOv8 Live Feed", frame)

        # Allow manual exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Manual exit requested. Closing camera...")
            break

    print("Captured 5 images. Exiting...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
