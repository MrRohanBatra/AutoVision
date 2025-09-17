import threading
import cv2
import numpy as np
from queue import Queue

from sympy import false
from ultralytics import YOLO
from sort.sort import Sort
from utils.util import save_detected_car
from utils.db_helper import add_detection

class CameraWorker(threading.Thread):
    def __init__(self, camera, plate_model_path, ocr_func, notify_queue):
        super().__init__()
        self.camera = camera
        self.plate_model = YOLO(plate_model_path,verbose=false)
        self.ocr_func = ocr_func
        self.notify_queue = notify_queue
        self.vehicle_model = YOLO("weights/yolov8n.pt",verbose=false)  # pre-trained YOLOv8n
        self.tracker = Sort()
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.camera.getCamera())
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            # Detect vehicles
            results = self.vehicle_model(frame)
            dets = []
            for det in results[0].boxes:
                cls_id = int(det.cls[0])
                if cls_id in [2,3,5,7]:  # car, truck, bus, motorcycle
                    x1,y1,x2,y2 = map(int, det.xyxy[0])
                    dets.append([x1,y1,x2,y2,1.0])

            if len(dets)==0:
                continue

            # Update SORT tracker
            dets_np = np.array(dets)
            tracked_objs = self.tracker.update(dets_np)

            for x1,y1,x2,y2,track_id in tracked_objs:
                vehicle_crop = frame[int(y1):int(y2), int(x1):int(x2)]

                # Detect plate
                plate_results = self.plate_model(vehicle_crop)
                for plate_det in plate_results[0].boxes:
                    px1,py1,px2,py2 = map(int, plate_det.xyxy[0])
                    plate_crop = vehicle_crop[py1:py2, px1:px2]
                    plate_number = self.ocr_func(plate_crop)

                    img_path = save_detected_car(vehicle_crop, plate_number, self.camera.getLocation())
                    add_detection(plate_number, self.camera.getLocation(), img_path)

                    # Send to notification queue
                    self.notify_queue.put((plate_number, img_path, self.camera.getLocation()))

        cap.release()

    def stop(self):
        self.running = False
