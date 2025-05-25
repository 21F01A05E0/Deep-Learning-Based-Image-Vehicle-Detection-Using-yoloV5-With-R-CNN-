from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
import cv2
import torch
import numpy as np
from PIL import Image
import os
from datetime import datetime
from detection import VehicleDetector
import threading
import time
import logging
from queue import Queue
from collections import deque
import queue
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure Flask app
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DEBUG'] = True  # Enable debug mode
app.config['TESTING'] = False  # Disable testing mode

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize vehicle detector
detector = VehicleDetector()
camera = None
detection_active = False
latest_detection = None
last_save_time = datetime.now()
detection_queue = Queue(maxsize=10)  # Queue for detection results
detection_history = deque(maxlen=10)  # Store recent detections for averaging

# Add these global variables after the existing ones
cumulative_counts = {
    'total': 0,
    'car': 0,
    'truck': 0,
    'bus': 0,
    'motorcycle': 0
}

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120))
    detection_results = db.relationship('DetectionResult', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Detection Result model
class DetectionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    total_vehicles = db.Column(db.Integer, nullable=False)
    cars = db.Column(db.Integer, nullable=False)
    trucks = db.Column(db.Integer, nullable=False)
    buses = db.Column(db.Integer, nullable=False)
    motorcycles = db.Column(db.Integer, nullable=False)
    accident_detected = db.Column(db.Boolean, default=False)
    detection_type = db.Column(db.String(20), nullable=False)  # 'live', 'image', or 'video'

class LatestCounts(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)
    total_vehicles = db.Column(db.Integer, nullable=False, default=0)
    cars = db.Column(db.Integer, nullable=False, default=0)
    trucks = db.Column(db.Integer, nullable=False, default=0)
    buses = db.Column(db.Integer, nullable=False, default=0)
    motorcycles = db.Column(db.Integer, nullable=False, default=0)
    accident_detected = db.Column(db.Boolean, default=False)
    last_updated = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

# Add this model after the LatestCounts model
class VideoProcessingResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    video_filename = db.Column(db.String(255), nullable=False)
    processed_video_path = db.Column(db.String(255), nullable=False)
    total_vehicles = db.Column(db.Integer, nullable=False)
    cars = db.Column(db.Integer, nullable=False)
    trucks = db.Column(db.Integer, nullable=False)
    buses = db.Column(db.Integer, nullable=False)
    motorcycles = db.Column(db.Integer, nullable=False)
    accident_detected = db.Column(db.Boolean, default=False)
    duration = db.Column(db.Float, nullable=False)
    fps = db.Column(db.Integer, nullable=False)
    processed_frames = db.Column(db.Integer, nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('signup'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('signup'))

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            if not next_page or not next_page.startswith('/'):
                next_page = url_for('dashboard')
            return redirect(next_page)
        else:
            flash('Invalid username or password')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get recent detection results from all types
    recent_detections = DetectionResult.query.filter_by(user_id=current_user.id)\
        .order_by(DetectionResult.timestamp.desc())\
        .limit(5)\
        .all()
    
    # Get recent video processing results
    recent_videos = VideoProcessingResult.query.filter_by(user_id=current_user.id)\
        .order_by(VideoProcessingResult.timestamp.desc())\
        .limit(5)\
        .all()
    
    # Combine and sort all activities by timestamp
    all_activities = []
    
    # Add detection results
    for result in recent_detections:
        all_activities.append({
            'type': result.detection_type,
            'timestamp': result.timestamp,
            'total_vehicles': result.total_vehicles,
            'cars': result.cars,
            'trucks': result.trucks,
            'buses': result.buses,
            'motorcycles': result.motorcycles,
            'accident_detected': result.accident_detected,
            'id': result.id
        })
    
    # Add video processing results
    for video in recent_videos:
        all_activities.append({
            'type': 'video',
            'timestamp': video.timestamp,
            'total_vehicles': video.total_vehicles,
            'cars': video.cars,
            'trucks': video.trucks,
            'buses': video.buses,
            'motorcycles': video.motorcycles,
            'accident_detected': video.accident_detected,
            'video_filename': video.video_filename,
            'duration': video.duration,
            'fps': video.fps,
            'id': video.id
        })
    
    # Sort all activities by timestamp, most recent first
    all_activities.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Limit to 10 most recent activities
    recent_activities = all_activities[:10]
    
    return render_template('dashboard.html', recent_activities=recent_activities)

@app.route('/live-detection')
@login_required
def live_detection():
    # Get recent live detection results
    recent_results = DetectionResult.query.filter_by(
        user_id=current_user.id,
        detection_type='live'
    ).order_by(DetectionResult.timestamp.desc()).limit(10).all()  # Increased limit to 10
    
    # Initialize camera if not already initialized
    camera = get_camera()
    if camera is None:
        flash('Failed to initialize camera. Please try again.')
    
    return render_template('live_detection.html', recent_results=recent_results)

@app.route('/video-upload', methods=['GET', 'POST'])
@login_required
def video_upload():
    if request.method == 'POST':
        if 'video' not in request.files:
            flash('No video file uploaded')
            return redirect(request.url)
            
        video = request.files['video']
        if video.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if video and allowed_file(video.filename, {'mp4', 'mov', 'avi'}):
            # Save the video file
            filename = secure_filename(video.filename)
            # Remove any existing extension
            base_filename = os.path.splitext(filename)[0]
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video.save(video_path)
            
            # Process the video
            try:
                # Open the video file
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    flash('Error opening video file')
                    return redirect(request.url)
                
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                logger.info(f"Processing video: {filename}")
                logger.info(f"Original FPS: {fps}, Total frames: {frame_count}, Duration: {duration:.1f}s")
                
                # Process frames at 5 FPS
                target_fps = 5
                frame_interval = int(fps / target_fps)
                processed_frames = 0
                total_processed = 0
                
                # Initialize counters for the video
                video_counts = {
                    'car': 0,
                    'motorcycle': 0,
                    'bus': 0,
                    'truck': 0,
                    'total': 0
                }
                
                # Create video writer for processed video
                processed_video_path = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{base_filename}.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H.264 codec
                out = cv2.VideoWriter(processed_video_path, fourcc, target_fps, (width, height))
                
                if not out.isOpened():
                    logger.error("Failed to create output video writer")
                    flash('Error creating output video')
                    return redirect(request.url)
                
                # Process first frame for preview
                ret, frame = cap.read()
                if ret:
                    # Save preview frame
                    preview_path = os.path.join(app.config['UPLOAD_FOLDER'], f'preview_{base_filename}.jpg')
                    cv2.imwrite(preview_path, frame)
                    
                    # Process frames at 5 FPS
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        total_processed += 1
                        if total_processed % frame_interval == 0:
                            processed_frames += 1
                            logger.info(f"Processing frame {processed_frames}")
                            
                            # Process the frame for detection
                            processed_frame, frame_counts, _ = detector.detect_vehicles(frame)
                            
                            # Convert to grayscale for accident detection
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                            edges = cv2.Canny(blurred, 50, 150)
                            
                            # Calculate edge density
                            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                            
                            # Look for irregular patterns
                            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            irregular_shapes = 0
                            for contour in contours:
                                area = cv2.contourArea(contour)
                                perimeter = cv2.arcLength(contour, True)
                                if area > 100:  # Filter small contours
                                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                                    if circularity < 0.5:  # Irregular shapes have low circularity
                                        irregular_shapes += 1
                            
                            # Calculate accident score
                            accident_score = 0
                            
                            # Score based on edge density
                            if edge_density > 0.15:
                                accident_score += 1
                            
                            # Score based on irregular shapes
                            if irregular_shapes > 5:
                                accident_score += 1
                            
                            # Score based on vehicle proximity
                            if frame_counts['total'] >= 2:
                                accident_score += 1
                            
                            # Final accident detection decision
                            accident_detected = accident_score >= 2
                            
                            # Update video counts
                            for vehicle_type in frame_counts:
                                if frame_counts[vehicle_type] > video_counts[vehicle_type]:
                                    video_counts[vehicle_type] = frame_counts[vehicle_type]
                            
                            # Add detection information to frame
                            info_text = f"Vehicles: {video_counts['total']} | "
                            info_text += f"Cars: {video_counts['car']} | "
                            info_text += f"Trucks: {video_counts['truck']} | "
                            info_text += f"Buses: {video_counts['bus']} | "
                            info_text += f"Motorcycles: {video_counts['motorcycle']}"
                            
                            if accident_detected:
                                info_text += " | ACCIDENT DETECTED!"
                                # Create overlay for accident visualization
                                overlay = processed_frame.copy()
                                cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
                                cv2.putText(overlay, "ACCIDENT DETECTED!", (10, 30),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                # Blend with original processed frame
                                processed_frame = cv2.addWeighted(overlay, 0.7, processed_frame, 0.3, 0)
                            
                            cv2.putText(processed_frame, info_text, (10, 60),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # Write processed frame to output video
                            out.write(processed_frame)
                            
                            # Save the last processed frame as preview
                            processed_preview_path = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_preview_{base_filename}.jpg')
                            cv2.imwrite(processed_preview_path, processed_frame)
                    
                    # Release video capture and writer
                    cap.release()
                    out.release()
                    
                    logger.info(f"Processed {processed_frames} frames at {target_fps} FPS")
                    logger.info(f"Final detection results: {video_counts}")
                    
                    # Save processing results to database
                    try:
                        result = VideoProcessingResult(
                            user_id=current_user.id,
                            video_filename=filename,
                            processed_video_path=f'processed_{base_filename}.mp4',
                            total_vehicles=video_counts['total'],
                            cars=video_counts['car'],
                            trucks=video_counts['truck'],
                            buses=video_counts['bus'],
                            motorcycles=video_counts['motorcycle'],
                            accident_detected=accident_detected,
                            duration=duration,
                            fps=target_fps,
                            processed_frames=processed_frames
                        )
                        db.session.add(result)
                        db.session.commit()
                        logger.info(f"Saved video processing results to database for user {current_user.id}")
                    except Exception as e:
                        logger.error(f"Error saving video processing results to database: {str(e)}")
                        db.session.rollback()
                    
                    return jsonify({
                        'success': True,
                        'preview_url': url_for('static', filename=f'uploads/preview_{base_filename}.jpg'),
                        'processed_preview_url': url_for('static', filename=f'uploads/processed_preview_{base_filename}.jpg'),
                        'processed_video_url': url_for('static', filename=f'uploads/processed_{base_filename}.mp4'),
                        'vehicle_counts': video_counts,
                        'accident_detected': accident_detected,
                        'accident_score': accident_score,
                        'edge_density': float(edge_density),
                        'irregular_shapes': irregular_shapes,
                        'duration': f'{duration:.1f}s',
                        'fps': f'{target_fps}',
                        'processed_frames': processed_frames
                    })
                
                cap.release()
                out.release()
                flash('Error processing video')
                return redirect(request.url)
                
            except Exception as e:
                logger.error(f"Error processing video: {str(e)}")
                flash('Error processing video')
                return redirect(request.url)
        else:
            flash('Invalid file type')
            return redirect(request.url)
            
    return render_template('video_upload.html')

@app.route('/video-history')
@login_required
def video_history():
    # Get video processing history for the current user
    video_results = VideoProcessingResult.query.filter_by(user_id=current_user.id)\
        .order_by(VideoProcessingResult.timestamp.desc())\
        .all()
    return render_template('video_history.html', video_results=video_results)

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/image-upload', methods=['GET', 'POST'])
@login_required
def image_upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image file uploaded')
            return redirect(request.url)
            
        image = request.files['image']
        if image.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if image and allowed_file(image.filename, {'jpg', 'jpeg', 'png'}):
            try:
                # Save detection result to database first to get the ID
                result = DetectionResult(
                    user_id=current_user.id,
                    total_vehicles=0,  # Will update after processing
                    cars=0,
                    trucks=0,
                    buses=0,
                    motorcycles=0,
                    accident_detected=False,
                    detection_type='image'
                )
                db.session.add(result)
                db.session.commit()
                
                # Now use the ID to create consistent filenames
                original_filename = f'image_{result.id}.jpg'
                processed_filename = f'processed_image_{result.id}.jpg'
                
                # Save the original image
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
                image.save(image_path)
                
                # Read and process the image
                frame = cv2.imread(image_path)
                if frame is None:
                    flash('Error reading image file')
                    return redirect(request.url)
                
                # Process the image for detection
                processed_frame, frame_counts, accident_detected = detector.detect_vehicles(frame)
                
                # Save the processed image
                processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
                cv2.imwrite(processed_image_path, processed_frame)
                
                # Update the detection result with actual counts
                result.total_vehicles = frame_counts['total']
                result.cars = frame_counts['car']
                result.trucks = frame_counts['truck']
                result.buses = frame_counts['bus']
                result.motorcycles = frame_counts['motorcycle']
                result.accident_detected = accident_detected
                db.session.commit()
                
                return jsonify({
                    'success': True,
                    'original_url': url_for('static', filename=f'uploads/{original_filename}'),
                    'processed_url': url_for('static', filename=f'uploads/{processed_filename}'),
                    'vehicle_counts': frame_counts,
                    'accident_detected': accident_detected
                })
                
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                flash('Error processing image')
                return redirect(request.url)
        else:
            flash('Invalid file type')
            return redirect(request.url)
            
    return render_template('image_upload.html')

@app.route('/image-history')
@login_required
def image_history():
    # Get image processing history for the current user
    image_results = DetectionResult.query.filter_by(
        user_id=current_user.id,
        detection_type='image'
    ).order_by(DetectionResult.timestamp.desc()).all()
    return render_template('image_history.html', image_results=image_results)

def get_camera():
    global camera
    if camera is None:
        try:
            # Release any existing camera first
            if camera is not None:
                camera.release()
            
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                logger.error("Failed to open camera")
                return None
                
            # Set camera properties for better performance
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
            camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
            camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Disable auto exposure
            
            # Wait for camera to initialize
            time.sleep(1)
            
            # Test camera read
            ret, frame = camera.read()
            if not ret:
                logger.error("Failed to read from camera")
                camera.release()
                camera = None
                return None
                
            logger.info("Camera initialized successfully")
            return camera
            
        except Exception as e:
            logger.error(f"Error initializing camera: {str(e)}")
            if camera is not None:
                camera.release()
            camera = None
            return None
    return camera

def save_detection_result(vehicle_counts, accident_detected):
    try:
        # Save the current cumulative counts directly
        result = DetectionResult(
            user_id=current_user.id,
            total_vehicles=vehicle_counts['total'],
            cars=vehicle_counts['car'],
            trucks=vehicle_counts['truck'],
            buses=vehicle_counts['bus'],
            motorcycles=vehicle_counts['motorcycle'],
            accident_detected=accident_detected,
            detection_type='live'
        )
        db.session.add(result)

        # Update or create latest counts
        latest_counts = LatestCounts.query.filter_by(user_id=current_user.id).first()
        if latest_counts is None:
            latest_counts = LatestCounts(user_id=current_user.id)
            db.session.add(latest_counts)

        latest_counts.total_vehicles = vehicle_counts['total']
        latest_counts.cars = vehicle_counts['car']
        latest_counts.trucks = vehicle_counts['truck']
        latest_counts.buses = vehicle_counts['bus']
        latest_counts.motorcycles = vehicle_counts['motorcycle']
        latest_counts.accident_detected = accident_detected
        latest_counts.last_updated = datetime.utcnow()

        db.session.commit()
        logger.info(f"Detection result saved: {vehicle_counts}")
    except Exception as e:
        logger.error(f"Error saving detection result: {str(e)}")
        db.session.rollback()

def db_worker():
    """Background worker to handle database updates"""
    while True:
        try:
            # Get detection result from queue
            vehicle_counts, accident_detected = detection_queue.get(timeout=1)
            
            # Save to database
            save_detection_result(vehicle_counts, accident_detected)
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error in db_worker: {str(e)}")
            time.sleep(1)

# Start database worker thread
db_thread = threading.Thread(target=db_worker, daemon=True)
db_thread.start()

def generate_frames():
    global detection_active, latest_detection, last_save_time, cumulative_counts, camera
    camera = get_camera()
    if camera is None:
        logger.error("No camera available")
        return
    
    frame_count = 0
    last_frame_time = time.time()
    frame_skip = 2  # Process every nth frame
    
    while True:
        try:
            # Add timeout for frame reading
            current_time = time.time()
            if current_time - last_frame_time > 5:  # 5 seconds timeout
                logger.error("Camera read timeout")
                break
                
            success, frame = camera.read()
            if not success:
                logger.error("Failed to read frame")
                break
                
            last_frame_time = current_time
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % frame_skip != 0:
                continue
                
            display_frame = frame.copy()
            
            if detection_active:
                try:
                    # Perform detection
                    processed_frame, vehicle_counts, accident_detected = detector.detect_vehicles(frame)
                    display_frame = processed_frame
                    
                    # Update cumulative counts
                    for vehicle_type in vehicle_counts:
                        if vehicle_counts[vehicle_type] > cumulative_counts[vehicle_type]:
                            cumulative_counts[vehicle_type] = vehicle_counts[vehicle_type]
                    
                    # Update latest detection results with cumulative counts
                    latest_detection = (cumulative_counts, accident_detected)
                    
                    # Add detection information to frame
                    info_text = f"Vehicles: {cumulative_counts['total']} | "
                    info_text += f"Cars: {cumulative_counts['car']} | "
                    info_text += f"Trucks: {cumulative_counts['truck']} | "
                    info_text += f"Buses: {cumulative_counts['bus']} | "
                    info_text += f"Motorcycles: {cumulative_counts['motorcycle']}"
                    
                    if accident_detected:
                        info_text += " | ACCIDENT DETECTED!"
                        cv2.putText(display_frame, "ACCIDENT DETECTED!", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    cv2.putText(display_frame, info_text, (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Save detection results periodically
                    current_time = datetime.now()
                    if (current_time - last_save_time).total_seconds() >= 5:
                        try:
                            save_detection_result(cumulative_counts, accident_detected)
                        except Exception as e:
                            logger.error(f"Error saving detection result: {str(e)}")
                        last_save_time = current_time
                        
                except Exception as e:
                    logger.error(f"Error in detection processing: {str(e)}")
            
            # Encode and yield frame with reduced quality
            ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                logger.error("Failed to encode frame")
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
        except Exception as e:
            logger.error(f"Error in generate_frames: {str(e)}")
            break
            
    # Cleanup
    if camera is not None:
        camera.release()
        camera = None
    logger.info("Camera stream ended")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection')
@login_required
def start_detection():
    global detection_active
    try:
        detection_active = True
        logger.info("Detection started")
        return jsonify({'status': 'success', 'message': 'Detection started'})
    except Exception as e:
        logger.error(f"Error starting detection: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop_detection')
@login_required
def stop_detection():
    global detection_active, latest_detection, detection_history, camera
    try:
        detection_active = False
        # Save final detection result if available
        if latest_detection:
            vehicle_counts, accident_detected = latest_detection
            try:
                save_detection_result(vehicle_counts, accident_detected)
            except Exception as e:
                logger.error(f"Error saving final detection result: {str(e)}")
        detection_history.clear()  # Clear detection history
        
        # Release camera
        if camera is not None:
            camera.release()
            camera = None
            
        logger.info("Detection stopped and camera released")
        return jsonify({'status': 'success', 'message': 'Detection stopped'})
    except Exception as e:
        logger.error(f"Error stopping detection: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/reset_counts')
@login_required
def reset_counts():
    global cumulative_counts
    try:
        # Reset global counts
        cumulative_counts = {
            'total': 0,
            'car': 0,
            'truck': 0,
            'bus': 0,
            'motorcycle': 0
        }
        
        # Reset database counts
        latest_counts = LatestCounts.query.filter_by(user_id=current_user.id).first()
        if latest_counts:
            latest_counts.total_vehicles = 0
            latest_counts.cars = 0
            latest_counts.trucks = 0
            latest_counts.buses = 0
            latest_counts.motorcycles = 0
            latest_counts.accident_detected = False
            latest_counts.last_updated = datetime.utcnow()
            db.session.commit()
        
        return jsonify({'status': 'success', 'message': 'Counts reset successfully'})
    except Exception as e:
        logger.error(f"Error resetting counts: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get_detection_stats')
@login_required
def get_detection_stats():
    try:
        # First try to get from latest_detection (active detection)
        if latest_detection:
            vehicle_counts, accident_detected = latest_detection
            return jsonify({
                'total_vehicles': vehicle_counts['total'],
                'cars': vehicle_counts['car'],
                'trucks': vehicle_counts['truck'],
                'buses': vehicle_counts['bus'],
                'motorcycles': vehicle_counts['motorcycle'],
                'accident_detected': accident_detected,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # If no active detection, get from database
        latest_counts = LatestCounts.query.filter_by(user_id=current_user.id).first()
        if latest_counts:
            return jsonify({
                'total_vehicles': latest_counts.total_vehicles,
                'cars': latest_counts.cars,
                'trucks': latest_counts.trucks,
                'buses': latest_counts.buses,
                'motorcycles': latest_counts.motorcycles,
                'accident_detected': latest_counts.accident_detected,
                'timestamp': latest_counts.last_updated.strftime('%Y-%m-%d %H:%M:%S')
            })
    except Exception as e:
        logger.error(f"Error getting detection stats: {str(e)}")
    
    return jsonify({
        'total_vehicles': 0,
        'cars': 0,
        'trucks': 0,
        'buses': 0,
        'motorcycles': 0,
        'accident_detected': False,
        'timestamp': None
    })

@app.route('/get_detection_history')
@login_required
def get_detection_history():
    try:
        recent_results = DetectionResult.query.filter_by(
            user_id=current_user.id,
            detection_type='live'
        ).order_by(DetectionResult.timestamp.desc()).limit(10).all()
        
        results = []
        for result in recent_results:
            results.append({
                'timestamp': result.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'total_vehicles': result.total_vehicles,
                'cars': result.cars,
                'trucks': result.trucks,
                'buses': result.buses,
                'motorcycles': result.motorcycles,
                'accident_detected': result.accident_detected
            })
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error getting detection history: {str(e)}")
        return jsonify([])

@app.route('/accident-detection', methods=['GET', 'POST'])
@login_required
def accident_detection():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image file uploaded')
            return redirect(request.url)
            
        image = request.files['image']
        if image.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if image and allowed_file(image.filename, {'jpg', 'jpeg', 'png'}):
            try:
                # Save detection result to database first to get the ID
                result = DetectionResult(
                    user_id=current_user.id,
                    total_vehicles=0,
                    cars=0,
                    trucks=0,
                    buses=0,
                    motorcycles=0,
                    accident_detected=False,
                    detection_type='accident'
                )
                db.session.add(result)
                db.session.commit()
                
                # Now use the ID to create consistent filenames
                original_filename = f'accident_{result.id}.jpg'
                processed_filename = f'processed_accident_{result.id}.jpg'
                
                # Save the original image
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
                image.save(image_path)
                
                # Read and process the image
                frame = cv2.imread(image_path)
                if frame is None:
                    flash('Error reading image file')
                    return redirect(request.url)

                # Convert to grayscale for additional processing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Apply Gaussian blur to reduce noise
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                
                # Edge detection using Canny
                edges = cv2.Canny(blurred, 50, 150)
                
                # Additional image processing for accident detection
                # 1. Detect strong edges (potential damage)
                edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                
                # 2. Look for irregular patterns
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                irregular_shapes = 0
                for contour in contours:
                    # Calculate contour properties
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    if area > 100:  # Filter small contours
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity < 0.5:  # Irregular shapes have low circularity
                            irregular_shapes += 1
                
                # Process the image for vehicle detection
                processed_frame, vehicle_counts, _ = detector.detect_vehicles(frame)
                
                # Accident detection criteria
                accident_detected = False
                accident_score = 0
                
                # Score based on edge density (more edges might indicate damage)
                if edge_density > 0.15:  # Threshold for edge density
                    accident_score += 1
                
                # Score based on irregular shapes (potential debris or damage)
                if irregular_shapes > 5:  # Threshold for irregular shapes
                    accident_score += 1
                
                # Score based on vehicle proximity
                if vehicle_counts['total'] >= 2:  # Multiple vehicles might indicate accident
                    accident_score += 1
                
                # Final accident detection decision
                accident_detected = accident_score >= 2
                
                # Draw accident detection visualization
                if accident_detected:
                    # Add red overlay to indicate accident areas
                    overlay = processed_frame.copy()
                    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
                    # Add text indicating accident
                    cv2.putText(overlay, "ACCIDENT DETECTED", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # Blend with original processed frame
                    processed_frame = cv2.addWeighted(overlay, 0.7, processed_frame, 0.3, 0)
                
                # Save the processed image
                processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
                cv2.imwrite(processed_image_path, processed_frame)
                
                # Update the detection result with actual counts and accident status
                result.total_vehicles = vehicle_counts['total']
                result.cars = vehicle_counts['car']
                result.trucks = vehicle_counts['truck']
                result.buses = vehicle_counts['bus']
                result.motorcycles = vehicle_counts['motorcycle']
                result.accident_detected = accident_detected
                db.session.commit()
                
                return jsonify({
                    'success': True,
                    'original_url': url_for('static', filename=f'uploads/{original_filename}'),
                    'processed_url': url_for('static', filename=f'uploads/{processed_filename}'),
                    'vehicle_counts': vehicle_counts,
                    'accident_detected': accident_detected,
                    'accident_score': accident_score,
                    'edge_density': float(edge_density),
                    'irregular_shapes': irregular_shapes
                })
                
            except Exception as e:
                logger.error(f"Error processing accident image: {str(e)}")
                flash('Error processing image')
                return redirect(request.url)
        else:
            flash('Invalid file type')
            return redirect(request.url)
            
    return render_template('accident_detection.html')

@app.route('/accident-history')
@login_required
def accident_history():
    # Get accident detection history for the current user
    accident_results = DetectionResult.query.filter_by(
        user_id=current_user.id,
        detection_type='accident'
    ).order_by(DetectionResult.timestamp.desc()).all()
    return render_template('accident_history.html', accident_results=accident_results)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    try:
        app.run(host='0.0.0.0', port=5001, debug=True)
    finally:
        # Cleanup camera on application exit
        if camera is not None:
            camera.release()
            camera = None 