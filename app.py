from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import base64
import socketio
import threading
import time
app = Flask(__name__)
sio = socketio.Server(async_mode='threading')
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)
# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)  # Use 0 for default camera, or 1, 2 etc for external.


def generate_frames():
    try:
        while True:
            success, frame = camera.read()  # Read frame
            if not success:
                print("Error reading frame!")
                break
            frame = cv2.flip(frame, 1) #flip camera so its mirrored

            #Process image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks and skeleton
            try:
               if results.pose_landmarks:
                   mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                               mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                               mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                               )
                   encoded_image = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode('utf-8')
                   sio.emit('video_frame', encoded_image)

               else:
                  encoded_image = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode('utf-8')
                  sio.emit('video_frame', encoded_image)

            except Exception as e:
                print(e)
    finally:
        camera.release()


@app.route('/')
def index():
    return render_template('index.html')

@sio.on('connect')
def connect(sid, environ):
    print('Client connected:', sid)
    global video_thread
    if 'video_thread' not in globals() or video_thread is None:
        video_thread = threading.Thread(target=generate_frames)
        video_thread.start()



@sio.on('disconnect')
def disconnect(sid):
    print('Client disconnected:', sid)
    global video_thread
    if 'video_thread' in globals() and video_thread is not None:
      video_thread = None

if __name__ == '__main__':
     app.run(debug=True, host='0.0.0.0')