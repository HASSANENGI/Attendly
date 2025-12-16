import os
import time
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from pinecone import Pinecone
from dotenv import load_dotenv

# Load the API key from the .env file
load_dotenv()
pinecone_api_key = os.getenv('PINECONE_API_KEY')

if pinecone_api_key is None:
    raise ValueError("PINECONE_API_KEY not found in .env file")

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)
index_name = "face-embeddings"
index = pc.Index(index_name)

# Load the YOLOv8 model for face detection
yolo_model = YOLO(r'Models\yolov8n_100e.pt')

# Load the pre-trained ResNet50 model for embeddings
resnet_model = models.resnet50(pretrained=True)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])  # Remove final classification layer
resnet_model.eval()

# Define preprocessing transformations for ResNet50
preprocess = transforms.Compose([
    transforms.Resize(256),  # Resize to 256x256
    transforms.CenterCrop(224),  # Crop to 224x224
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

# Function to generate an embedding for a single face
def generate_embedding_from_face(face_image):
    # Convert the face image to PIL format for preprocessing
    face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess(face_pil).unsqueeze(0)  # Add batch dimension

    # Move to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_model.to(device)
    input_tensor = input_tensor.to(device)

    # Generate the embedding
    with torch.no_grad():
        features = resnet_model(input_tensor)  # Shape: (1, 2048, 1, 1)
        embedding = features.squeeze(-1).squeeze(-1).cpu().numpy()  # Shape: (2048,)

    return embedding

# Function to query Pinecone for the most similar embedding
def find_most_similar(embedding):
    # Query Pinecone for the most similar embeddings
    query_results = index.query(vector=embedding.tolist(), top_k=1, include_values=True)
    
    if query_results["matches"]:
        best_match = query_results["matches"][0]
        best_match_name = best_match["id"]
        best_match_similarity = best_match["score"]
        return best_match_name, best_match_similarity
    else:
        return "Unknown", 0.0

# Main function for live face detection, embedding generation, and comparison
def main():
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Camera is live. Showing feed with bounding boxes.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        # Run YOLOv8 for face detection
        results = yolo_model.predict(source=frame, verbose=False)
        
        for result in results:
            for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
                if conf >= 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = map(int, box[:4])
                    cropped_face = frame[y1:y2, x1:x2]

                    # Draw bounding box and confidence score
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Conf: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    # Ensure valid face crop
                    if cropped_face.size > 0:
                        # Generate embedding for the detected face
                        embedding = generate_embedding_from_face(cropped_face)

                        # Find the most similar embedding in Pinecone
                        best_match_name, similarity = find_most_similar(embedding)

                        # Display the match on the frame
                        match_text = f"{best_match_name} ({similarity:.2f})"
                        cv2.putText(frame, match_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Display the frame with bounding boxes and match information
        cv2.imshow("YOLOv8 Face Recognition", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Manual exit requested. Closing camera...")
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
