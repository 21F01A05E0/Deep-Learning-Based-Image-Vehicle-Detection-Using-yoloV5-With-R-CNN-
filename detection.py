import torch
import cv2
import numpy as np
from PIL import Image
import os
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import time

class AccidentDetector(nn.Module):
    def __init__(self):
        super(AccidentDetector, self).__init__()
        # Load pre-trained ResNet model
        self.model = models.resnet50(pretrained=True)
        # Modify the final layer for binary classification (accident/no accident)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # 2 classes: accident and no accident
        )
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        # Reshape input from [batch, frames, channels, height, width] to [batch * frames, channels, height, width]
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(-1, channels, height, width)  # Reshape to [batch * frames, channels, height, width]
        
        # Process each frame through the model
        outputs = self.model(x)
        
        # Reshape back to [batch, frames, num_classes]
        outputs = outputs.view(batch_size, num_frames, -1)
        
        # Average predictions across frames
        outputs = outputs.mean(dim=1)  # [batch, num_classes]
        
        return outputs

class VehicleDetector:
    def __init__(self):
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model.conf = 0.25  # Lower confidence threshold for more detections
        self.model.iou = 0.45
        
        # Vehicle classes we're interested in
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        # Initialize counters
        self.vehicle_counts = {
            'car': 0,
            'motorcycle': 0,
            'bus': 0,
            'truck': 0,
            'total': 0
        }
        
        # Initialize accident detector
        self.accident_detector = AccidentDetector()
        self.accident_detector.eval()  # Set to evaluation mode
        
        # Accident detection parameters
        self.accident_threshold = 0.7
        self.accident_detected = False
        self.accident_frames = 0
        self.accident_frames_threshold = 3
        self.previous_detections = []
        
        # Load pre-trained weights if available
        try:
            self.accident_detector.load_state_dict(torch.load('accident_model.pth'))
            print("Loaded pre-trained accident detection model")
        except:
            print("No pre-trained model found. Using default weights.")
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.accident_detector.to(self.device)
        
        # Initialize frame buffer for accident detection
        self.frame_buffer = []
        self.frame_buffer_size = 8
        
        # Performance optimization parameters
        self.frame_skip = 1  # Process every frame
        self.frame_counter = 0
        self.last_accident_check = 0
        self.accident_check_interval = 0.5

    def detect_vehicles(self, frame):
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame for faster processing while maintaining aspect ratio
        height, width = frame_rgb.shape[:2]
        max_size = 640
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
        
        # Perform vehicle detection
        results = self.model(frame_rgb)
        
        # Process detections
        current_detections = []
        frame_counts = {k: 0 for k in self.vehicle_counts}  # Counts for current frame only
        
        # Debug print for all detections
        print("\nDetections in frame:")
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            cls = int(cls)
            print(f"Class: {cls}, Confidence: {conf:.2f}")
            
            if cls in self.vehicle_classes:
                vehicle_type = self.vehicle_classes[cls]
                frame_counts[vehicle_type] += 1
                frame_counts['total'] += 1
                
                current_detections.append({
                    'box': [x1, y1, x2, y2],
                    'class': vehicle_type,
                    'confidence': conf
                })
                
                # Draw bounding box with confidence
                # Scale coordinates back to original frame size
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{vehicle_type} {conf:.2f}', 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        print(f"Current frame counts: {frame_counts}")
        
        # Update frame buffer
        self.frame_buffer.append(frame_rgb)
        if len(self.frame_buffer) > self.frame_buffer_size:
            self.frame_buffer.pop(0)
        
        # Check for accidents periodically
        current_time = time.time()
        if current_time - self.last_accident_check >= self.accident_check_interval:
            self.check_for_accidents()
            self.last_accident_check = current_time
        
        # Update previous detections
        self.previous_detections = current_detections
        
        return frame, frame_counts, self.accident_detected

    def check_for_accidents(self):
        if len(self.frame_buffer) < self.frame_buffer_size:
            return

        try:
            # Prepare frames for accident detection
            frames = []
            for frame in self.frame_buffer:
                # Convert to PIL Image and resize for faster processing
                frame_pil = Image.fromarray(frame).resize((224, 224))
                # Apply transformations
                frame_tensor = self.accident_detector.transform(frame_pil)
                frames.append(frame_tensor)
            
            # Stack frames and add batch dimension
            frames_tensor = torch.stack(frames).unsqueeze(0)  # [1, frames, channels, height, width]
            frames_tensor = frames_tensor.to(self.device)
            
            # Perform inference
            with torch.no_grad():
                outputs = self.accident_detector(frames_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                accident_prob = probabilities[0][1].item()  # Probability of accident
                
                print(f"Accident probability: {accident_prob:.2f}")
                
                # Update accident detection status
                if accident_prob > self.accident_threshold:
                    self.accident_frames += 1
                    if self.accident_frames >= self.accident_frames_threshold:
                        self.accident_detected = True
                        print("Accident detected!")
                else:
                    self.accident_frames = max(0, self.accident_frames - 1)
                    if self.accident_frames == 0:
                        self.accident_detected = False
                        
        except Exception as e:
            print(f"Error in accident detection: {str(e)}")
            self.accident_detected = False

    def calculate_iou(self, box1, box2):
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union 