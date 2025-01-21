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

def get_relevant_keypoints(keypoints):
    points = {}
    if keypoints:
         for keypoint in keypoints:
              if keypoint.HasField('name') and keypoint.name in [
                'left_shoulder', 'right_shoulder',
                'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist',
                'left_hip', 'right_hip',
                'nose'
            ]:
                  points[keypoint.name] = keypoint
    return points

def draw_doll(keypoints, width, height):
    doll_canvas = np.zeros((height, width, 3), dtype=np.uint8)
    doll_width = width * 0.8
    doll_height = height * 0.7
    head_radius = doll_width / 8
    head_x = width // 2
    head_y = int(height * 0.2)
    body_top_y = head_y + head_radius
    #Draw head
    if 'nose' in keypoints:
        nose = keypoints['nose']
        if nose.visibility > 0.5:
           cv2.circle(doll_canvas, (head_x, head_y), int(head_radius), (245, 207, 177), -1)
           cv2.circle(doll_canvas, (head_x, head_y), int(head_radius), (0,0,0), 2)

    #Draw body
    if ('left_shoulder' in keypoints and
        'right_shoulder' in keypoints and
        'left_hip' in keypoints and
        'right_hip' in keypoints):

        left_shoulder = keypoints['left_shoulder']
        right_shoulder = keypoints['right_shoulder']
        left_hip = keypoints['left_hip']
        right_hip = keypoints['right_hip']
        if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5 and left_hip.visibility > 0.5 and right_hip.visibility > 0.5:
           body_bottom_y = body_top_y + int(doll_height)
           cv2.line(doll_canvas, (head_x, body_top_y), (head_x, body_bottom_y), (168, 67, 0), 6)

         # Draw arms
        if ('left_elbow' in keypoints and
            'left_wrist' in keypoints):
              left_shoulder = keypoints['left_shoulder']
              left_elbow = keypoints['left_elbow']
              left_wrist = keypoints['left_wrist']
              if left_shoulder.visibility > 0.5 and left_elbow.visibility > 0.5 and left_wrist.visibility > 0.5:
                left_shoulder_x = head_x - (doll_width//2)
                left_elbow_x = int(left_shoulder_x + (left_elbow.x - left_shoulder.x) / (640 / (doll_width/2)) * -1)
                left_elbow_y = int(body_top_y+ ((left_elbow.y -left_shoulder.y) / (480/ (doll_height/2))) * 1)

                left_wrist_x = int(left_elbow_x + (left_wrist.x - left_elbow.x) / (640 / (doll_width/2)) * -1)
                left_wrist_y = int(left_elbow_y + ((left_wrist.y -left_elbow.y) / (480/ (doll_height/2))) * 1)

                cv2.line(doll_canvas, (left_shoulder_x, body_top_y), (left_elbow_x, left_elbow_y), (168, 67, 0), 6)
                cv2.line(doll_canvas, (left_elbow_x, left_elbow_y), (left_wrist_x, left_wrist_y), (168, 67, 0), 6)

        if ('right_elbow' in keypoints and
            'right_wrist' in keypoints):
                right_shoulder = keypoints['right_shoulder']
                right_elbow = keypoints['right_elbow']
                right_wrist = keypoints['right_wrist']
                if right_shoulder.visibility > 0.5 and right_elbow.visibility > 0.5 and right_wrist.visibility > 0.5:
                   right_shoulder_x = head_x + (doll_width // 2)
                   right_elbow_x = int(right_shoulder_x + (right_elbow.x - right_shoulder.x) / (640 / (doll_width / 2)) * -1)
                   right_elbow_y = int(body_top_y + ((right_elbow.y - right_shoulder.y) / (480 / (doll_height / 2))) * 1)

                   right_wrist_x = int(right_elbow_x + (right_wrist.x - right_elbow.x) / (640 / (doll_width / 2)) * -1)
                   right_wrist_y = int(right_elbow_y + ((right_wrist.y - right_elbow.y) / (480 / (doll_height / 2))) * 1)

                   cv2.line(doll_canvas, (right_shoulder_x, body_top_y), (right_elbow_x, right_elbow_y), (168, 67, 0), 6)
                   cv2.line(doll_canvas, (right_elbow_x, right_elbow_y), (right_wrist_x, right_wrist_y), (168, 67, 0), 6)

    #Draw legs
    if 'left_hip' in keypoints and 'right_hip' in keypoints:
        left_hip = keypoints['left_hip']
        right_hip = keypoints['right_hip']
        if left_hip.visibility > 0.5 and right_hip.visibility > 0.5:
              body_bottom_y = body_top_y + int(doll_height)
              left_hip_x = head_x - (doll_width // 2)
              right_hip_x = head_x + (doll_width // 2)
              cv2.line(doll_canvas, (left_hip_x, body_bottom_y), (left_hip_x, body_bottom_y + 50), (168, 67, 0), 6)
              cv2.line(doll_canvas, (right_hip_x, body_bottom_y), (right_hip_x, body_bottom_y + 50), (168, 67, 0), 6)

    return doll_canvas

def generate_frames():
    try:
        while True:
            success, frame = camera.read()  # Read frame
            if not success:
                print("Error reading frame!")
                break
            frame = cv2.flip(frame, 1) #flip camera so its mirrored
            height, width, _ = frame.shape

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
                   keypoints = get_relevant_keypoints(results.pose_landmarks.landmark)
                   doll_canvas = draw_doll(keypoints, width//2 , height//2)

                   encoded_image = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode('utf-8')
                   encoded_doll = base64.b64encode(cv2.imencode('.jpg', doll_canvas)[1]).decode('utf-8')

                   sio.emit('video_frame', encoded_image)
                   sio.emit('doll_frame', encoded_doll)
               else:
                  encoded_image = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode('utf-8')
                  encoded_doll = base64.b64encode(cv2.imencode('.jpg', np.zeros((height//2, width//2, 3), dtype=np.uint8))[1]).decode('utf-8')
                  sio.emit('video_frame', encoded_image)
                  sio.emit('doll_frame', encoded_doll)


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
    app.run(debug=True, port=1100)