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
import time
import numpy as np
import tempfile
import threading
import queue
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, RedirectResponse
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
model_loaded = False
model_error = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model_loaded, model_error
    print("\n" + "="*60)
    print("🚀 YOLO TRACKING SERVICE STARTING...")
    print("="*60)
    print(f"[STARTUP] Device: {device}")
    print(f"[STARTUP] Loading YOLO model on startup...")
    
    try:
        load_yolo_model()
        model_loaded = True
        print(f"[STARTUP] ✅ YOLO model loaded successfully!")
        print("="*60 + "\n")
    except Exception as e:
        model_error = str(e)
        print(f"[STARTUP] ❌ FAILED TO LOAD YOLO MODEL: {e}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")


class TrackingConfig(BaseModel):
    """Tracking configuration"""
    tracking_method: str = "bytetrack"
    reid_model: Optional[str] = None
    confidence: float = 0.5
    iou_threshold: float = 0.7
    class_filter: Optional[list] = None


def load_yolo_model():
    """Load YOLO model (cached) - auto-download if needed"""
    global yolo_model
    if yolo_model is None:
        print("[YOLO] Loading model: yolov8n.pt")
        try:
            # Try loading from local file first
            yolo_model = YOLO("yolov8n.pt")
            print("[YOLO] Model loaded successfully!")
        except Exception as e:
            print(f"[YOLO] Error loading local model: {e}")
            print("[YOLO] Attempting to download model from Ultralytics...")
            try:
                # Downloads from Ultralytics Hub if not found locally
                yolo_model = YOLO("yolov8n.pt")
                print("[YOLO] Model downloaded and loaded successfully!")
            except Exception as e2:
                print(f"[YOLO] CRITICAL ERROR: Could not load or download model: {e2}")
                raise
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


def process_frame_with_tracking(frame, tracker, yolo, config: TrackingConfig, frame_num=0):
    """Process a single frame for tracking"""
    if frame is None:
        print(f"[PROCESS] Frame {frame_num}: ERROR - frame is None")
        return frame, []
    
    if frame.size == 0:
        print(f"[PROCESS] Frame {frame_num}: ERROR - frame is empty")
        return frame, []
    
    # Run YOLO inference
    try:
        results = yolo(frame, conf=config.confidence, iou=config.iou_threshold, verbose=False)
        result = results[0]
        
        # Get detections
        dets = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        
        if frame_num % 50 == 0 or frame_num <= 3:
            print(f"[YOLO] Frame {frame_num}: Detected {len(dets)} objects | Conf threshold: {config.confidence} | Frame shape: {frame.shape}")
            if len(dets) > 0 and frame_num <= 3:
                print(f"[YOLO] Frame {frame_num}: First detection - box: {dets[0]} conf: {confs[0]} class: {classes[0]}")
    
    except Exception as e:
        if frame_num % 50 == 0:
            print(f"[PROCESS] Frame {frame_num}: ERROR in YOLO inference: {e}")
        return frame, []
    
    # Filter by class if specified
    if config.class_filter:
        mask = np.isin(classes, config.class_filter)
        dets = dets[mask]
        confs = confs[mask]
        classes = classes[mask]
    
    # Update tracker
    try:
        if len(dets) > 0:
            # Format: [x1, y1, x2, y2, confidence, class_id]
            dets_confs = np.concatenate([
                dets,
                confs.reshape(-1, 1),
                classes.reshape(-1, 1)
            ], axis=1)
            
            if frame_num <= 3:
                print(f"[TRACKER] Frame {frame_num}: Input format check - shape: {dets_confs.shape}, first row: {dets_confs[0]}")
            
            tracks = tracker.update(dets_confs, frame)
            
            if frame_num <= 3 or frame_num % 50 == 0:
                print(f"[TRACKER] Frame {frame_num}: Got {len(tracks)} tracks from {len(dets)} detections")
                if len(tracks) > 0 and frame_num <= 3:
                    print(f"[TRACKER] Frame {frame_num}: First track - {tracks[0]}")
        else:
            tracks = np.array([])
            if frame_num <= 3:
                print(f"[TRACKER] Frame {frame_num}: No detections, skipping tracking")
    
    except Exception as e:
        if frame_num % 50 == 0:
            print(f"[PROCESS] Frame {frame_num}: ERROR in tracker update: {e}")
        tracks = np.array([])
    
    # Draw results
    try:
        annotated_frame = frame.copy()
        if len(tracks) > 0:
            for track in tracks:
                x1, y1, x2, y2, track_id = track[:5].astype(int)
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw label
                label = f"ID: {int(track_id)}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    except Exception as e:
        if frame_num % 50 == 0:
            print(f"[PROCESS] Frame {frame_num}: ERROR drawing boxes: {e}")
        annotated_frame = frame.copy()
    
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
    """Root endpoint - redirect to camera interface"""
    return RedirectResponse(url="/camera")


@app.get("/camera")
async def camera_page():
    """Serve the camera tracking web interface"""
    with open("index.html", "r") as f:
        return FileResponse("index.html", media_type="text/html")


@app.get("/health")
async def health():
    """Health check endpoint"""
    health_status = {
        "status": "healthy" if model_loaded else "degraded",
        "device": device,
        "model_loaded": model_loaded,
        "streaming": stream_active
    }
    
    if model_error:
        health_status["model_error"] = model_error
    
    if not model_loaded:
        health_status["status"] = "unhealthy"
        health_status["message"] = f"YOLO model not loaded: {model_error or 'Unknown error'}"
    
    return health_status


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
    client_addr = websocket.client.host if websocket.client else "unknown"
    print(f"\n[WS] New connection attempt from {client_addr}")
    
    await websocket.accept()
    print(f"[WS] Connection accepted from {client_addr}")
    
    # Check if model is loaded
    if not model_loaded:
        print(f"[WS] ❌ ERROR: YOLO model not loaded! Error: {model_error}")
        try:
            await websocket.send_json({
                "error": f"YOLO model not loaded. Server error: {model_error}"
            })
        except:
            pass
        await websocket.close()
        return
    
    tracker = None
    yolo = None
    frame_num = 0
    config = None
    
    try:
        # Wait for initial config
        print(f"[WS] Waiting for config from {client_addr}...")
        while True:
            try:
                data = await websocket.receive_text()
                print(f"[WS] Received text data from {client_addr}: {data[:100]}")
                
                msg = json.loads(data)
                if msg.get("type") == "config":
                    tracking_method = msg.get("tracking_method", "bytetrack")
                    confidence = msg.get("confidence", 0.5)
                    
                    print(f"[WS] Config received: method={tracking_method}, conf={confidence}")
                    
                    if tracking_method not in ALLOWED_TRACKERS:
                        error_msg = f"Invalid tracker: {tracking_method}"
                        print(f"[WS] Error: {error_msg}")
                        await websocket.send_json({"error": error_msg})
                        continue
                    
                    # Initialize tracker and model
                    print(f"[WS] Loading YOLO model...")
                    yolo = load_yolo_model()
                    print(f"[WS] Creating tracker: {tracking_method}")
                    tracker = create_tracker_instance(tracking_method)
                    config = TrackingConfig(
                        tracking_method=tracking_method,
                        confidence=confidence
                    )
                    
                    print(f"[WS] Tracker ready! Sending confirmation to {client_addr}")
                    await websocket.send_json({
                        "status": "ready",
                        "type": "config_response",
                        "tracker": tracking_method,
                        "message": "Ready to receive frames"
                    })
                    break
            except json.JSONDecodeError as e:
                print(f"[WS] JSON decode error: {e}")
                continue
            except Exception as e:
                print(f"[WS] Error during config: {e}")
                raise
        
        # Process incoming frames
        print(f"[WS] Starting frame processing from {client_addr}")
        frame_count = 0
        expected_seq_num = 0
        dropped_frames = 0
        last_sent_time = time.time()
        processing_times = []
        
        while True:
            try:
                # Receive frame data (binary with sequence number prefix)
                frame_data = await websocket.receive_bytes()
                frame_count += 1
                recv_time = time.time()
                
                # Extract sequence number from first 4 bytes
                if len(frame_data) < 4:
                    continue
                
                seq_num = int.from_bytes(frame_data[:4], byteorder='little', signed=False)
                jpeg_data = frame_data[4:]
                
                # Check sequence order - skip if out of order
                if seq_num != expected_seq_num:
                    dropped_frames += 1
                    if frame_count % 50 == 0:
                        print(f"[WS] Frame {frame_count}: Skipped out-of-order frame (seq={seq_num}, expected={expected_seq_num})")
                    expected_seq_num = seq_num + 1
                    continue
                
                expected_seq_num = seq_num + 1
                
                # Decode frame from JPEG
                decode_start = time.time()
                frame = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
                decode_time = time.time() - decode_start
                
                if frame is None:
                    continue
                
                # Process frame with tracking - THIS DRAWS THE BOXES
                process_start = time.time()
                annotated_frame, tracks_data = process_frame_with_tracking(frame, tracker, yolo, config, frame_count)
                process_time = time.time() - process_start
                frame_num += 1
                
                # Encode result as JPEG (binary, no base64)
                encode_start = time.time()
                ret, frame_buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                encode_time = time.time() - encode_start
                
                if not ret:
                    continue
                
                # Send response as binary: [seq_num(4)] [object_count(4)] [jpeg_data]
                send_start = time.time()
                response = bytearray()
                response.extend(seq_num.to_bytes(4, byteorder='little', signed=False))
                object_count = len(tracks_data)
                response.extend(object_count.to_bytes(4, byteorder='little', signed=False))
                response.extend(frame_buffer.tobytes())
                
                try:
                    await websocket.send_bytes(bytes(response))
                    send_time = time.time() - send_start
                    
                    if frame_num <= 3:
                        print(f"[SEND] Frame {seq_num}: Response sent! Size={len(response)} bytes, ObjectCount={object_count}, ResponseTime={send_time*1000:.0f}ms")
                    
                except Exception as send_err:
                    print(f"[SEND] Frame {seq_num}: ERROR sending response - {send_err}")
                    raise
                
                total_time = time.time() - recv_time
                last_sent_time = time.time()
                
                processing_times.append(process_time)
                if len(processing_times) > 50:
                    processing_times.pop(0)
                
                if frame_count % 50 == 0:
                    avg_process = sum(processing_times) / len(processing_times)
                    print(f"[WS] Frame {seq_num}: Total={total_time*1000:.0f}ms (decode={decode_time*1000:.0f}ms, process={avg_process*1000:.0f}ms, encode={encode_time*1000:.0f}ms, send={send_time*1000:.0f}ms) | Objects: {len(tracks_data)} | Dropped: {dropped_frames}")
                
            except WebSocketDisconnect:
                print(f"[WS] Client {client_addr} disconnected after {frame_count} frames (dropped: {dropped_frames})")
                break
            except Exception as e:
                print(f"[WS] Frame {frame_count}: ERROR - {e}")
                try:
                    await websocket.send_json({"error": str(e)})
                except:
                    pass
                break
    
    except Exception as e:
        print(f"[WS] WebSocket error from {client_addr}: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        try:
            await websocket.close()
            print(f"[WS] Connection closed from {client_addr}")
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
