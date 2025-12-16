# Attendance System – Face Recognition

An end-to-end face recognition pipeline for attendance marking. The app detects faces from a webcam using YOLOv8, generates robust embeddings with ResNet50, stores them in a Pinecone vector index, and performs live identification by nearest-neighbor search.

## Features
- Face detection with YOLOv8 (fast and accurate bounding boxes).
- Embedding generation using ResNet50 (2048-d feature vectors).
- Pinecone vector database integration for scalable similarity search.
- Simple new-user enrollment flow (capture + store embedding).
- Live recognition with on-screen identity and similarity score.
- Environment-based configuration with `.env`.

## Repository Layout
- `new_user.py` — Capture 5 cropped face images for a new user into `cropped_faces/<UserName>/`.
- `embedding.py` — Average the folder’s embeddings and upsert to Pinecone (`index: face-embeddings`). Also supports capture like `new_user.py`.
- `face_recognition.py` — Live detection + embedding + Pinecone query; displays best match.
- `check_embedding.py` — Print Pinecone index stats to verify stored vectors.
- `cropped_faces/` — Local per-user image folders (5 crops each recommended).
- `models/` — YOLOv8 weights (`yolov8n_100e.pt`, `yolov8m_200e.pt`, `yolov8l_100e.pt`).
- `requirements.txt` — Python dependencies.

## Setup
1) Python 3.10+ recommended.
2) Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3) Create `.env` at the project root:
```
PINECONE_API_KEY=<your-api-key>
```

4) Confirm YOLO weights path. Scripts default to `Models\yolov8n_100e.pt`; if your weights are in `models\yolov8n_100e.pt`, pass `-w models\yolov8n_100e.pt`.

## Usage
- Enroll a new user (save local crops):
```powershell
python new_user.py -w models\yolov8n_100e.pt -t 0.5
```
Enter the user/folder name when prompted.

- Generate and store the user’s embedding in Pinecone:
```powershell
python embedding.py -w models\yolov8n_100e.pt -t 0.5
```
Provide the same folder name; the script averages embeddings and upserts to `face-embeddings`.

- Run live recognition:
```powershell
python face_recognition.py
```

- Inspect the Pinecone index:
```powershell
python check_embedding.py
```

## How it works
1. Face detection: YOLOv8 finds face bounding boxes in frames.
2. Preprocessing: Crops are resized/cropped and normalized via torchvision.
3. Embeddings: ResNet50 (classifier head removed) produces a 2048-d vector; folders use the mean vector.
4. Storage & Query: Pinecone serverless index stores vectors by user id; recognition queries nearest neighbor and shows name + similarity.

## Notes
- Embedding dimension: 2048 (ResNet50 pooled features).
- Pinecone index: `face-embeddings` (serverless, AWS `us-east-1`).
- Press `q` to exit any OpenCV window.
- If CUDA is available, PyTorch uses it automatically.

## Roadmap
- Config file for thresholds/paths.
- Attendance logging (CSV/DB) with timestamps.
- Batch enrollment, updates, and deletion tools.
- Basic GUI.

---

© Muhammad Hassan. All rights reserved.
