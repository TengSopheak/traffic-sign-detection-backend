from inference_sdk import InferenceHTTPClient
from inference import InferencePipeline
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import tempfile
import threading
import time
import config
from collections import defaultdict

app = FastAPI()

# Allow frontend (React Vite) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Change to your Vite frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=config.roboflow_api
)

@app.post("/upload-image") # for testing change "app" to "upload_image_router"
async def upload_image(file: UploadFile = File(...)):
    # 1. Save uploaded file to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_file.write(await file.read())
    temp_file.close()

    try:
        # Perform inference using the uploaded image
        result = client.run_workflow(
            workspace_name="orpf",
            workflow_id="detect-and-visualize",
            images={"image": temp_file.name},
            use_cache=True
        )

        # Return the result as JSON
        return JSONResponse(content=result)

    finally:
        # Cleanup temp file
        os.unlink(temp_file.name)

# Shared results list
video_results = []
video_results_lock = threading.Lock()

# Helper to serialize objects that are not JSON serializable
def serialize_result(result, frame_index=0):
    try:
        if not result or "predictions" not in result:
            return {"frame": frame_index, "predictions": []}

        detections = result["predictions"]
        if len(detections) == 0:
            return {"frame": frame_index, "predictions": []}
        
        detections = result["predictions"]
        boxes = detections.xyxy  # NumPy array of [x1, y1, x2, y2]
        confidences = detections.confidence
        classes = detections.data["class_name"]
        class_ids = detections.class_id
        image_dims = detections.data["image_dimensions"][0]  # [width, height]

        serialized_predictions = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            width = float(x2 - x1)
            height = float(y2 - y1)
            x_center = float(x1 + width / 2)
            y_center = float(y1 + height / 2)

            serialized_predictions.append({
                "class": str(classes[i]),
                "class_id": int(class_ids[i]),
                "confidence": float(confidences[i]),
                "bbox": {
                    "x": x_center,
                    "y": y_center,
                    "width": width,
                    "height": height
                },
                "image_width": int(image_dims[0]),
                "image_height": int(image_dims[1])
            })

        return {
            "frame": frame_index,
            "predictions": serialized_predictions
        }

    except Exception as e:
        print("Serialization error:", e)
        return None



# Prediction callback
def my_sink(result, video_frame):
    frame_index = getattr(video_frame, "frame_id", 0)
    serializable_result = serialize_result(result, frame_index)
    if serializable_result:
        with video_results_lock:
            video_results.append(serializable_result)

# -----------------------
# VIDEO FILE UPLOAD
# -----------------------
@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    file_extension = os.path.splitext(file.filename)[-1].lower()
    if file_extension not in [".mp4", ".mov", ".mkv"]:
        return JSONResponse(content={"error": "Unsupported file format"}, status_code=400)

    # Save uploaded video to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name

    try:
        # Clear previous results
        with video_results_lock:
            video_results.clear()

        # Initialize pipeline for this specific video
        pipeline = InferencePipeline.init_with_workflow(
            api_key=config.roboflow_api,
            workspace_name="orpf",
            workflow_id="detect-and-visualize",
            video_reference=temp_path,
            max_fps=30,
            on_prediction=my_sink
        )

        # Start and wait for it to complete
        pipeline.start()
        pipeline.join()

        # Return all collected predictions
        with video_results_lock:
            results_copy = list(video_results)

        # Group predictions by frame index
        grouped_results = defaultdict(list)
        for item in results_copy:
            frame = item["frame"]
            grouped_results[frame].extend(item.get("predictions", []))

        # Construct final output
        merged_results = []
        for frame, predictions in grouped_results.items():
            merged_results.append({
                "frame": frame,
                "predictions": predictions
            })

        return JSONResponse(content={"message": "Video processed", "results": merged_results})

    finally:
        # Wait briefly to ensure OS has released the file
        time.sleep(0.5)
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except PermissionError:
                time.sleep(0.5)
                os.unlink(temp_path)