### SafeZone Predictor with YOLO

SafeZone Predictor is a FastAPI-based application that utilizes the YOLO (You Only Look Once) model for object detection. The application allows users to upload images, processes them using the YOLO model, and returns the detected objects with bounding boxes drawn on the images.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/faisal-fida/SafeZone-Predictor-Yolo.git
   cd SafeZone-Predictor-Yolo
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the YOLO model weights:**
   Ensure that the `best.pt` file is in the root directory of the project.

## Usage

1. **Run the FastAPI application:**
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Access the API:**
   Open your browser and navigate to `http://127.0.0.1:8000/docs` to access the Swagger UI for the API.

## API Endpoints

### POST /predict/

**Description:** Upload an image file to predict objects in the image.

**Request:**
- **file**: The image file to be uploaded.

**Response:**
- **image**: The path to the output image with bounding boxes.
- **predictions**: The classes of the detected objects.

Example using `curl`:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@your_image.jpg'
```
