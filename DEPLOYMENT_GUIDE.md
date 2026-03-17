# YOLO Tracking Web Service - Deployment Guide

## Overview
This is a FastAPI-based web service for real-time YOLO object detection and tracking. It supports multiple tracking methods and can be deployed to Railway, AWS, GCP, or any container-based platform.

## Quick Start - Local Development

### Prerequisites
- Docker & Docker Compose
- or Python 3.11+

### Option 1: Using Docker (Recommended)

```bash
# Build and run with docker-compose
docker-compose up --build

# API will be available at http://localhost:8000
```

### Option 2: Using Python Virtual Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-web.txt

# Run the server
python main.py
```

## API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation.

### Endpoints

#### 1. **Health Check**
```bash
GET /health
```
Response:
```json
{
  "status": "healthy",
  "device": "cuda" or "cpu"
}
```

#### 2. **Get Available Trackers**
```bash
GET /trackers
```
Response:
```json
{
  "available_trackers": ["bytetrack", "botsort", "deepocsort", "ocsort", "strongsort", "hybridsort"],
  "default": "bytetrack"
}
```

#### 3. **Upload Video/Image for Tracking**
```bash
POST /track/upload?tracking_method=bytetrack&confidence=0.5&iou_threshold=0.7
Content-Type: multipart/form-data

Body: file (video or image)
```

**Query Parameters:**
- `tracking_method`: One of `bytetrack`, `botsort`, `deepocsort`, `ocsort`, `strongsort`, `hybridsort` (default: `bytetrack`)
- `confidence`: Detection confidence threshold (0.0-1.0, default: 0.5)
- `iou_threshold`: IoU threshold for NMS (0.0-1.0, default: 0.7)

**Response:**
```json
{
  "message": "Processing complete",
  "frames_processed": 300,
  "output_video": "/path/to/tracked_video.mp4",
  "total_tracks": 15,
  "tracking_data": [
    {
      "frame": 0,
      "tracks": [
        {
          "track_id": 1,
          "bbox": [x1, y1, x2, y2],
          "confidence": 0.95
        }
      ]
    }
  ]
}
```

#### 4. **Real-time WebSocket Streaming** (Coming Soon)
```
WS /track/stream
```

#### 5. **Download Processed Video**
```bash
GET /download/{filename}
```

## Deployment to Railway

### Prerequisites
1. Railway account (https://railway.app)
2. GitHub account with your project pushed

### Step-by-Step Deployment

1. **Push your project to GitHub**
   ```bash
   git push origin main
   ```

2. **Connect to Railway**
   - Go to https://railway.app
   - Click "New Project"
   - Select "Deploy from GitHub"
   - Connect your GitHub account
   - Select the `yaherd` repository

3. **Configure Environment**
   - Railway automatically detects the Dockerfile
   - Set environment variables if needed in Railway dashboard:
     - `PORT=8000` (usually set automatically)
     - `PYTHONUNBUFFERED=1`

4. **Deploy**
   - Click "Deploy Now"
   - Wait for build to complete (~5-15 minutes)
   - Railway will provide a public URL

5. **Access Your Service**
   - Railway provides a URL like: `https://yaherd-production.up.railway.app`
   - API docs: `https://your-railway-url/docs`
   - Health check: `https://your-railway-url/health`

## Example Usage

### Using Python requests
```python
import requests

# Upload file for tracking
with open('video.mp4', 'rb') as f:
    files = {'file': f}
    params = {
        'tracking_method': 'bytetrack',
        'confidence': 0.5
    }
    response = requests.post(
        'http://localhost:8000/track/upload',
        files=files,
        params=params
    )
    print(response.json())
```

### Using cURL
```bash
curl -X POST "http://localhost:8000/track/upload?tracking_method=botsort" \
  -F "file=@video.mp4"
```

### Using Python client library
```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.get('http://localhost:8000/trackers')
    print(response.json())
```

## Architecture

```
main.py (FastAPI App)
├── /track/upload → Video Processing
│   ├── Load YOLO model
│   ├── Create Tracker
│   ├── Process frames
│   └── Save annotated video
├── /track/stream → WebSocket Real-time
└── /download → File Retrieval
```

## Performance Tips

1. **GPU Support**
   - The service automatically uses CUDA if available
   - For GPU support on Railway, you may need a paid plan with GPU
   - Check Railway dashboard for GPU options

2. **Model Caching**
   - Models are cached in memory on first use
   - Subsequent requests are faster

3. **File Limits**
   - Be mindful of upload size limits
   - Large videos may timeout; consider splitting them

## Troubleshooting

### Issue: Module not found errors
**Solution:** Ensure all dependencies in `requirements-web.txt` are installed

### Issue: "Could not find model"
**Solution:** Models are downloaded on first use. Check disk space.

### Issue: Memory errors on Railway
**Solution:** Railway free tier has limited RAM. Consider:
- Using smaller YOLO models (yolov8n)
- Processing shorter videos
- Upgrading Railway plan

### Issue: Timeout on large video uploads
**Solution:** 
- Process shorter videos
- Increase Railway timeout settings
- Stream processing instead of batch processing

## Next Steps

1. **Add Authentication:** Implement JWT or API keys
2. **Add Database:** Store tracking history in PostgreSQL
3. **Add Queue System:** Use Redis + Celery for async processing
4. **Add Frontend:** Create web UI for easy access
5. **Multi-GPU:** Scale horizontally on Kubernetes

## Support

For issues or questions:
1. Check FastAPI docs: https://fastapi.tiangolo.com/
2. Check Railway docs: https://docs.railway.app/
3. Check YOLO docs: https://docs.ultralytics.com/
4. Check BoxMOT GitHub: https://github.com/mikel-brostrom/yolo_tracking
