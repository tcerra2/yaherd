@echo off
cd /d c:\Users\tcerr\Documents\Yolo\yolo_tracking-master\yolo_tracking-master
set PYTHONPATH=c:\Users\tcerr\Documents\Yolo\yolo_tracking-master\yolo_tracking-master;%PYTHONPATH%
C:\Users\tcerr\Documents\Yolo\.venv\Scripts\python.exe tracking/track.py --source 0 --tracking-method botsort --yolo-model yolov8n.pt --conf 0.5
pause
