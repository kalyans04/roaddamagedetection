import os
from flask import Flask, render_template, request, redirect, url_for, send_file
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/temp/'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}
MODEL_PATH = "/Users/kalyanshivanadhuni/Desktop/roaddamagedetection/YOLOv8_Small_RDD.pt"

# Ensure the temp directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load YOLO model
model = YOLO(MODEL_PATH)

# Classes for road damage detection
CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'video_input.mp4')
            file.save(filepath)
            confidence_threshold = float(request.form.get('confidence_threshold', 0.5))
            process_video(filepath, confidence_threshold)
            return redirect(url_for('download'))
    return render_template('index.html')

def process_video(input_path, confidence_threshold):
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'video_infer.mp4')
    video_capture = cv2.VideoCapture(input_path)
    
    if not video_capture.isOpened():
        return "Error opening video file"

    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Perform inference
        results = model.predict(frame, conf=confidence_threshold)
        
        # Draw bounding boxes
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                class_id = int(box.cls)
                label = CLASSES[class_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write the frame into the output file
        out.write(frame)

    # Release everything if job is finished
    video_capture.release()
    out.release()

@app.route('/download')
def download():
    return render_template('download.html')

@app.route('/download_video')
def download_video():
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], 'video_infer.mp4'), as_attachment=True, attachment_filename='RDD_Prediction.mp4')

if __name__ == '__main__':
    app.run(debug=True)
