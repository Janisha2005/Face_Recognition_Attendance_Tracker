# Face_Recognition_Attendance_Tracker

This project is a web-based face recognition attendance system built with Flask and OpenCV. 
It allows users to register their faces and mark attendance using live camera feeds. 
The admin can view attendance statistics for registered users.

## 🚀 Features

- Real-time face detection and recognition using OpenCV
- Register new users with their facial image
- Mark attendance based on face recognition
- Calculate attendance summary for each user
- Web interface with separate Admin and User views

## 📁 Project Structure

- ├── app.py # Main Flask application
- ├── face_recognition.py # Face recognition and attendance logic
- ├── templates/
- │ ├── index.html # Homepage
- │ ├── user.html # User interface for registration & marking attendance
- │ ├── admin.html # Admin interface to view attendance stats
- ├── static/
- │ └── style.css # CSS styling (you may add this if not yet created)
- ├── registered_faces/ # Folder to store registered face images
- ├── yearly_attendance.csv # CSV file storing attendance data
- ├── requirements.txt # Python dependencies
