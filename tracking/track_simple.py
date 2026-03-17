# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
import cv2
import numpy as np
from pathlib import Path
import torch
from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS
from ultralytics import YOLO


@torch.no_grad()
def run(args):
    """Run tracking on video/webcam using YOLOv8 detections."""
    
    # Load YOLOv8 model
    yolo = YOLO(
        args.yolo_model if 'yolov8' in str(args.yolo_model) else 'yolov8n.pt',
    )

    # Initialize tracker
    tracking_config = TRACKER_CONFIGS / (args.tracking_method + '.yaml')
    tracker = create_tracker(
        args.tracking_method,
        tracking_config,
        args.reid_model,
        device='cpu',
        half=False,
        per_class=args.per_class
    )
    if hasattr(tracker, 'model'):
        tracker.model.warmup()

    # Get predictions using predict mode (not track mode)
    results = yolo.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        show=False,
        stream=True,
        device=args.device,
        verbose=args.verbose,
        imgsz=args.imgsz,
        classes=args.classes,
    )

    # Process each frame with our tracker
    for result in results:
        # Extract detections
        if result.boxes is not None and len(result.boxes) > 0:
            # Get bounding boxes in xyxy format [x1, y1, x2, y2]
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy().reshape(-1, 1)
            clss = result.boxes.cls.cpu().numpy().reshape(-1, 1)
            
            # Create detection array [x1, y1, x2, y2, conf, cls]
            dets = np.hstack([boxes_xyxy, confs, clss])
        else:
            dets = np.empty((0, 6))

        # Update tracker with detections
        tracks = tracker.update(dets, result.orig_img)

        # Plot tracked results
        img = tracker.plot_results(result.orig_img, args.show_trajectories)

        # Display
        if args.show:
            cv2.imshow('BoxMOT - Object Tracking', img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') or key == ord('q'):
                break

    cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='botsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack, hybridsort')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video')
    parser.add_argument('--show-trajectories', action='store_true',
                        help='display object trajectories')
    parser.add_argument('--verbose', action='store_true',
                        help='verbose output')
    parser.add_argument('--save', action='store_true',
                        help='save video')
    parser.add_argument('--per-class', action='store_true',
                        help='use per-class trackers')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    
    opt = parser.parse_args()
    return opt


def main(opt):
    run(opt)


if __name__ == '__main__':
    opt = parse_opt()
    opt.show = True  # Enable display by default
    main(opt)
