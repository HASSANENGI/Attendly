import os
import time
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from ultralytics import YOLO
from pinecone import Pinecone, ServerlessSpec  # Updated import
from PIL import Image
from dotenv import load_dotenv  # Import dotenv to load environment variables from .env file
import argparse  # Import argparse for command-line argument parsing

# Load the API key from the .env file
load_dotenv()  # Load environment variables from the .env file
pinecone_api_key = os.getenv('PINECONE_API_KEY')  # Get the API key from the environment variable

if pinecone_api_key is None:
    raise ValueError("PINECONE_API_KEY not found in .env file")

# Initialize Pinecone client with new API
pc = Pinecone(api_key=pinecone_api_key)

# Create Pinecone index for storing embeddings (Dimension 2048 for ResNet50)
index_name = "face-embeddings"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=2048,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
index = pc.Index(index_name)

# Load the pre-trained ResNet50 model for generating embeddings
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final classification layer
model.eval()

# Define the preprocessing transformations for ResNet50
preprocess = transforms.Compose([
    transforms.Resize(256),  # Resize to 256x256
    transforms.CenterCrop(224),  # Crop to 224x224
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

# Function to load and preprocess images for embedding generation
def load_and_preprocess_images(image_folder):
    images = []
    for file_name in os.listdir(image_folder):
        if file_name.endswith(('.jpg', '.png', '.jpeg')):  # Only process image files
            img_path = os.path.join(image_folder, file_name)
            image = Image.open(img_path).convert('RGB')
            input_tensor = preprocess(image)
            images.append(input_tensor)
    return torch.stack(images)  # Stack tensors into a batch

# Function to generate a single embedding for a batch of images
def generate_single_embedding(image_folder):
    # Load and preprocess all images in the folder
    input_batch = load_and_preprocess_images(image_folder)
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_batch = input_batch.to(device)
    
    # Extract embeddings
    with torch.no_grad():  # Disable gradient computation
        features = model(input_batch)  # Shape: (batch_size, 2048, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # Remove extra dimensions: (batch_size, 2048)
    
    # Compute the average embedding
    average_embedding = features.mean(dim=0)  # Shape: (2048,)
    
    return average_embedding.cpu().numpy()  # Return as a NumPy array

# Function to store embedding in Pinecone
def save_embedding_to_pinecone(embedding, person_name):
    # Store the embedding in Pinecone
    index.upsert([(person_name, embedding)])  # Using person_name as the ID and embedding as values
    print(f"Saved embedding for {person_name} to Pinecone.")

# Function to parse command-line arguments
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

    # After capturing 5 images, generate the embedding and save it to Pinecone
    embedding = generate_single_embedding(output_folder)  # Generate the embedding from the saved images
    save_embedding_to_pinecone(embedding, folder_name)  # Save the embedding in Pinecone with the folder name as ID

if __name__ == '__main__':
    main()
