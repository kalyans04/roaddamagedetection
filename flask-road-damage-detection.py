# from flask import Flask, render_template, Response
# import cv2
# from ultralytics import YOLO
# import numpy as np

# app = Flask(__name__)

# # Load YOLO model
# model = YOLO('/Users/kalyanshivanadhuni/Desktop/roaddamagedetection/YOLOv8_Small_RDD.pt')

# # Classes
# CLASSES = [
#     "Longitudinal Crack",
#     "Transverse Crack",
#     "Alligator Crack",
#     "Potholes"
# ]

# def generate_frames():
#     camera = cv2.VideoCapture(0)  # Use 0 for webcam
#     while True:
#         success, frame = camera.read()
#         if not success:
#             break
#         else:
#             # Resize and process frame
#             frame_resized = cv2.resize(frame, (640, 640))
#             results = model(frame_resized)
            
#             # Draw bounding boxes
#             for result in results:
#                 boxes = result.boxes.cpu().numpy()
#                 for box in boxes:
#                     x1, y1, x2, y2 = box.xyxy[0].astype(int)
#                     class_id = int(box.cls)
#                     label = CLASSES[class_id]
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(frame, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')  # Just the filename, not the full path

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, Response, request
import cv2
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

# Load YOLO model
model = YOLO('/Users/kalyanshivanadhuni/Desktop/roaddamagedetection/YOLOv8_Small_RDD.pt')

# Classes
CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

# Set a default threshold
DEFAULT_THRESHOLD = 0.5

def generate_frames(threshold):
    camera = cv2.VideoCapture(0)  # Use 0 for webcam
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Resize and process frame
            frame_resized = cv2.resize(frame, (640, 640))
            results = model(frame_resized, conf=threshold)  # Apply threshold
            
            # Draw bounding boxes
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    class_id = int(box.cls)
                    label = CLASSES[class_id]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
def index():
    # Get the threshold value from the form, or use the default
    threshold = request.form.get('threshold', DEFAULT_THRESHOLD)
    threshold = float(threshold)
    return render_template('index.html', threshold=threshold)

@app.route('/video_feed')
def video_feed():
    threshold = request.args.get('threshold', DEFAULT_THRESHOLD, type=float)
    return Response(generate_frames(threshold), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
