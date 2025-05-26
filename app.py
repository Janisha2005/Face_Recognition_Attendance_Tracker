from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
import os
from datetime import datetime
import numpy as np
from face_recognition import registered_faces, mark_attendance, calculate_attendance_stats, recognize_face, add_name_to_attendance, initialize_yearly_attendance, registered_faces_dir

app = Flask(__name__)

# Initialize attendance
initialize_yearly_attendance()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Global variables for video capture
video_capture = cv2.VideoCapture(0)
output_frame = None
name="Unknown"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/user')
def user_page():
    return render_template('user.html')

@app.route('/admin')
def admin_page():
    return render_template('admin.html')


@app.route('/video_feed')
def video_feed():
    """Stream video feed to the browser."""
    def generate_frames():
        global output_frame
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                frame_array = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
                gray_frame = cv2.cvtColor(frame_array, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

                for (x, y, w, h) in faces:
                    face_roi = gray_frame[y:y+h, x:x+w]
                    name = recognize_face(face_roi)
                    cv2.rectangle(frame_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame_array, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            output_frame = frame_array.copy()
            _, encoded_image = cv2.imencode('.jpg', output_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   bytearray(encoded_image) +
                   b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['POST'])
def register():
    """Register a new face."""
    name = request.form.get('name').strip()
    if not name:
        return redirect(url_for('index1'))
    
    ret, frame = video_capture.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_to_register = gray_frame[y:y+h, x:x+w]
        file_path = os.path.join(registered_faces_dir, f"{name}.jpg")
        cv2.imwrite(file_path, face_to_register)
        message = add_name_to_attendance(name)
    else:
        message = "No face detected. Try again."
    return jsonify(message=message)

@app.route('/mark_attendance_route', methods=['POST'])
def mark_attendance_route():
    """Mark attendance for recognized faces."""
    global output_frame
    if output_frame is not None:
        gray_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        today_date = datetime.now().strftime("%Y-%m-%d")
        
        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y+h, x:x+w]
            name = recognize_face(face_roi)
            if name != "Unknown":
                message = mark_attendance(name, today_date)
    return jsonify(message=message)

@app.route('/calculate_attendance', methods=['POST'])
def calculate_attendance():
    name = request.form.get('name').strip()
    message = calculate_attendance_stats(name)
    return jsonify(message=message)

if __name__ == '__main__':
    app.run(debug=True)