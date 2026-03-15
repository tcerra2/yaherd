"""
YOLO Tracking Web Service - FastAPI Application
Supports both file uploads and real-time streaming
"""

import io
import os
import cv2
import torch
import numpy as np
import tempfile
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import WEIGHTS, TRACKER_CONFIGS
from ultralytics import YOLO

# Configuration
ALLOWED_TRACKERS = ["bytetrack", "botsort", "deepocsort", "ocsort", "strongsort", "hybridsort"]
UPLOAD_DIR = Path(tempfile.gettempdir()) / "yolo_tracking_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize FastAPI
app = FastAPI(
    title="YOLO Tracking Web Service",
    description="Real-time object detection and tracking API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model caching
yolo_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"


class TrackingConfig(BaseModel):
    """Tracking configuration"""
    tracking_method: str = "bytetrack"
    reid_model: Optional[str] = None
    confidence: float = 0.5
    iou_threshold: float = 0.7
    class_filter: Optional[list] = None


def load_yolo_model():
    """Load YOLO model (cached)"""
    global yolo_model
    if yolo_model is None:
        yolo_model = YOLO("yolov8n.pt")
    return yolo_model


def create_tracker_instance(tracking_method: str, reid_model: Optional[str] = None):
    """Create a tracker instance"""
    if tracking_method not in ALLOWED_TRACKERS:
        raise ValueError(f"Unsupported tracker: {tracking_method}. Allowed: {ALLOWED_TRACKERS}")
    
    reid_model = reid_model or str(WEIGHTS / 'osnet_x0_25_msmt17.pt')
    tracking_config = TRACKER_CONFIGS / (tracking_method + '.yaml')
    
    tracker = create_tracker(
        tracking_method,
        tracking_config,
        reid_model,
        device,
        half=False,
        per_class=False
    )
    return tracker


def process_frame(frame, tracker, yolo, config: TrackingConfig):
    """Process a single frame for tracking"""
    results = yolo(frame, conf=config.confidence, iou=config.iou_threshold, verbose=False)
    result = results[0]
    
    # Get detections
    dets = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    
    # Filter by class if specified
    if config.class_filter:
        mask = np.isin(classes, config.class_filter)
        dets = dets[mask]
        confs = confs[mask]
        classes = classes[mask]
    
    # Update tracker
    if len(dets) > 0:
        dets_confs = np.concatenate([dets, confs.reshape(-1, 1)], axis=1)
        tracks = tracker.update(dets_confs, frame)
    else:
        tracks = np.array([])
    
    # Draw results
    annotated_frame = frame.copy()
    if len(tracks) > 0:
        for track in tracks:
            x1, y1, x2, y2, track_id = track[:5].astype(int)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"ID: {int(track_id)}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Prepare response data
    tracks_data = []
    if len(tracks) > 0:
        for track in tracks:
            x1, y1, x2, y2, track_id = track[:5]
            tracks_data.append({
                "track_id": int(track_id),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(track[4]) if len(track) > 4 else 0.0
            })
    
    return annotated_frame, tracks_data


# Routes

@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "service": "YOLO Tracking Web Service",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "upload": "POST /track/upload",
            "stream": "WS /track/stream",
            "trackers": "GET /trackers"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "device": device}


@app.get("/trackers")
async def get_trackers():
    """Get list of available trackers"""
    return {
        "available_trackers": ALLOWED_TRACKERS,
        "default": "bytetrack"
    }


@app.post("/track/upload")
async def upload_video(
    file: UploadFile = File(...),
    tracking_method: str = Query("bytetrack"),
    confidence: float = Query(0.5),
    iou_threshold: float = Query(0.7)
):
    """Upload a video/image file for tracking"""
    
    if tracking_method not in ALLOWED_TRACKERS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid tracker. Allowed: {ALLOWED_TRACKERS}"}
        )
    
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Load models
        yolo = load_yolo_model()
        tracker = create_tracker_instance(tracking_method)
        
        config = TrackingConfig(
            tracking_method=tracking_method,
            confidence=confidence,
            iou_threshold=iou_threshold
        )
        
        # Process video
        cap = cv2.VideoCapture(str(file_path))
        all_tracks = []
        frame_count = 0
        
        output_path = UPLOAD_DIR / f"tracked_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, tracks_data = process_frame(frame, tracker, yolo, config)
            
            # Initialize video writer
            if out is None:
                h, w = annotated_frame.shape[:2]
                out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (w, h))
            
            out.write(annotated_frame)
            all_tracks.append({
                "frame": frame_count,
                "tracks": tracks_data
            })
            frame_count += 1
        
        cap.release()
        if out:
            out.release()
        
        return {
            "message": "Processing complete",
            "frames_processed": frame_count,
            "output_video": str(output_path),
            "total_tracks": len(all_tracks),
            "tracking_data": all_tracks[:10]  # Return first 10 frames
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.websocket("/track/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming"""
    await websocket.accept()
    
    try:
        # Load models
        yolo = load_yolo_model()
        
        while True:
            # Receive frame data
            data = await websocket.receive_json()
            
            if data.get("type") == "init":
                # Initialize tracker
                tracking_method = data.get("tracking_method", "bytetrack")
                tracker = create_tracker_instance(tracking_method)
                await websocket.send_json({"status": "initialized", "tracker": tracking_method})
            
            elif data.get("type") == "frame":
                # Process frame
                # In real implementation, receive frame bytes
                await websocket.send_json({"status": "processing"})
            
            elif data.get("type") == "close":
                break
    
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close()


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download processed video"""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        return JSONResponse(status_code=404, content={"error": "File not found"})
    return FileResponse(file_path)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
