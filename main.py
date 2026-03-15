"""
YOLO Tracking Web Service with Real-time Streaming
Live video display with object tracking - accessible from any device
"""

import io
import os
import cv2
import json
import base64
import torch
import numpy as np
import tempfile
import threading
import queue
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
    description="Real-time object detection and tracking API with live streaming",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
yolo_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
stream_queue = queue.Queue(maxsize=1)
stream_active = False
current_tracker = None


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


def process_frame_with_tracking(frame, tracker, yolo, config: TrackingConfig):
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


def video_stream_generator(video_path: str, tracking_method: str = "bytetrack", config_params: dict = None):
    """Generate MJPEG stream from video file - like cv2.imshow() but over web"""
    global stream_active, current_tracker
    
    config_params = config_params or {}
    config = TrackingConfig(tracking_method=tracking_method, **config_params)
    
    try:
        yolo = load_yolo_model()
        tracker = create_tracker_instance(tracking_method)
        current_tracker = tracker
        
        cap = cv2.VideoCapture(video_path)
        stream_active = True
        
        while stream_active and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize for faster processing
            frame = cv2.resize(frame, (640, 480))
            
            # Process frame
            annotated_frame, _ = process_frame_with_tracking(frame, tracker, yolo, config)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            # Yield MJPEG boundary and frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + f'{len(frame_bytes)}'.encode() + b'\r\n\r\n'
                   + frame_bytes + b'\r\n')
        
        cap.release()
        stream_active = False
    
    except Exception as e:
        print(f"Stream error: {e}")
        stream_active = False


# Routes

@app.get("/")
async def root():
    """Root endpoint with streaming instructions"""
    hostname = os.getenv("RAILWAY_PUBLIC_DOMAIN", "localhost:8000")
    return {
        "service": "YOLO Tracking Web Service v2",
        "version": "2.0.0",
        "features": ["Real-time live streaming", "File upload", "Live tracking"],
        "how_to_use": {
            "step_1": "Open web UI at: /camera",
            "step_2": "Allow camera access from your device",
            "step_3": "Click 'Start Tracking' to begin live tracking",
            "step_4": "See real-time tracked video instantly"
        },
        "endpoints": {
            "web_ui": "/camera",
            "health": "/health",
            "stream": "/stream?file=video.mp4&tracking_method=bytetrack",
            "upload": "POST /track/upload",
            "trackers": "GET /trackers",
            "stop_stream": "POST /stop-stream"
        }
    }


@app.get("/camera")
async def camera_page():
    """Serve the camera tracking web interface"""
    with open("index.html", "r") as f:
        return FileResponse("index.html", media_type="text/html")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "device": device, "streaming": stream_active}


@app.get("/trackers")
async def get_trackers():
    """Get list of available trackers"""
    return {
        "available_trackers": ALLOWED_TRACKERS,
        "default": "bytetrack"
    }


@app.get("/stream")
async def stream_video(
    file: str = Query(..., description="Filename of uploaded video"),
    tracking_method: str = Query("bytetrack"),
    confidence: float = Query(0.5),
    iou_threshold: float = Query(0.7)
):
    """Stream video with real-time tracking (MJPEG format) - Like cv2.imshow() but accessible from any device over web"""
    
    if tracking_method not in ALLOWED_TRACKERS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid tracker. Allowed: {ALLOWED_TRACKERS}"}
        )
    
    # Check if file exists
    video_path = UPLOAD_DIR / file
    
    if not video_path.exists():
        return JSONResponse(status_code=404, content={"error": f"Video file not found: {file}"})
    
    config_params = {
        "confidence": confidence,
        "iou_threshold": iou_threshold
    }
    
    return StreamingResponse(
        video_stream_generator(str(video_path), tracking_method, config_params),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/track/upload")
async def upload_video(
    file: UploadFile = File(...),
    tracking_method: str = Query("bytetrack"),
    confidence: float = Query(0.5),
    iou_threshold: float = Query(0.7)
):
    """Upload a video file and get real-time stream URL"""
    
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
        
        hostname = os.getenv("RAILWAY_PUBLIC_DOMAIN", "localhost:8000")
        
        return {
            "message": "Video uploaded! Now streaming...",
            "filename": file.filename,
            "stream_url": f"https://{hostname}/stream?file={file.filename}&tracking_method={tracking_method}&confidence={confidence}&iou_threshold={iou_threshold}",
            "how_to_view": "Open stream_url in a browser or img element to watch live tracking",
            "example_html": f'<img src="https://{hostname}/stream?file={file.filename}" width="640" height="480">',
            "available_trackers": ALLOWED_TRACKERS
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
        
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


@app.post("/stop-stream")
async def stop_stream():
    """Stop current streaming"""
    global stream_active
    stream_active = False
    return {"status": "stream stopped"}


@app.websocket("/ws/track")
async def websocket_track(websocket: WebSocket):
    """WebSocket endpoint for real-time camera tracking from client devices"""
    await websocket.accept()
    
    tracker = None
    yolo = None
    frame_num = 0
    config = None
    
    try:
        # Wait for initial config
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "config":
                    tracking_method = msg.get("tracking_method", "bytetrack")
                    confidence = msg.get("confidence", 0.5)
                    
                    if tracking_method not in ALLOWED_TRACKERS:
                        await websocket.send_json({"error": f"Invalid tracker: {tracking_method}"})
                        continue
                    
                    # Initialize tracker and model
                    yolo = load_yolo_model()
                    tracker = create_tracker_instance(tracking_method)
                    config = TrackingConfig(
                        tracking_method=tracking_method,
                        confidence=confidence
                    )
                    
                    await websocket.send_json({
                        "status": "ready",
                        "tracker": tracking_method,
                        "message": "Ready to receive frames"
                    })
                    break
            except json.JSONDecodeError:
                continue
        
        # Process incoming frames
        while True:
            try:
                # Receive frame data (binary)
                frame_data = await websocket.receive_bytes()
                
                # Decode frame from JPEG
                frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                # Process frame with tracking
                annotated_frame, tracks_data = process_frame_with_tracking(frame, tracker, yolo, config)
                frame_num += 1
                
                # Encode result as JPEG
                ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret:
                    continue
                
                # Convert to base64 for JSON transport
                frame_base64 = base64.b64encode(buffer).decode()
                
                # Send back to client
                await websocket.send_json({
                    "type": "frame",
                    "frame_data": f"data:image/jpeg;base64,{frame_base64}",
                    "frame_num": frame_num,
                    "object_count": len(tracks_data)
                })
                
            except WebSocketDisconnect:
                print("Client disconnected")
                break
            except Exception as e:
                print(f"Frame processing error: {e}")
                await websocket.send_json({"error": str(e)})
                break
    
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


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
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
