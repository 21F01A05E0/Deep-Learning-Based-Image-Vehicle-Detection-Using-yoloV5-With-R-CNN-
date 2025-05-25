# Deep Learning-Based Vehicle Image Detection Using YOLOv5 With Region-Based Convolutional Neural Network

This project implements a vehicle detection system using YOLOv5 and Region-Based Convolutional Neural Network (R-CNN) for accurate vehicle detection in images and videos. The system includes features for real-time detection, video analysis, and image analysis, with additional capabilities for accident detection.

## Features

- Real-time vehicle detection using webcam
- Video upload and analysis
- Image upload and analysis
- Accident detection and alert system
- Support for multiple vehicle types (cars, trucks, buses, motorcycles)
- Modern web interface with Tailwind CSS
- User authentication system

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for better performance)
- Webcam (for live detection)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vehicle-detection-system.git
cd vehicle-detection-system
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download the YOLOv5 model weights:
```bash
# Create a models directory
mkdir models
# Download the YOLOv5 weights (you'll need to provide the actual weights file)
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Create an account or log in to access the features.

## Project Structure

```
vehicle-detection-system/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── models/               # YOLOv5 model weights
├── static/              # Static files (CSS, JS, uploads)
│   └── uploads/         # Uploaded files storage
└── templates/           # HTML templates
    ├── base.html        # Base template
    ├── index.html       # Home page
    ├── login.html       # Login page
    ├── signup.html      # Signup page
    ├── dashboard.html   # Dashboard
    ├── live_detection.html  # Live detection page
    ├── video_upload.html    # Video upload page
    └── image_upload.html    # Image upload page
```

## API Endpoints

- `/` - Home page
- `/signup` - User registration
- `/login` - User login
- `/dashboard` - User dashboard
- `/live-detection` - Live vehicle detection
- `/video-upload` - Video upload and analysis
- `/image-upload` - Image upload and analysis
- `/video_feed` - Live video feed endpoint

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv5 by Ultralytics
- Flask web framework
- Tailwind CSS
- OpenCV
- PyTorch 