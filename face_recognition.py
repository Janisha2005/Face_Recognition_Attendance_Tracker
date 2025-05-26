import cv2
import os
import pandas as pd
import numpy as np

registered_faces_dir = "registered_faces"
attendance_file = "yearly_attendance.csv"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if not os.path.exists(registered_faces_dir):
    os.makedirs(registered_faces_dir)

def initialize_yearly_attendance():
    if not os.path.exists(attendance_file):
        registered_faces = load_registered_faces()
        df = pd.DataFrame({"Name": list(registered_faces.keys())})
        df.to_csv(attendance_file, index=False)

def load_registered_faces():
    registered_faces = {}
    for filename in os.listdir(registered_faces_dir):
        if filename.endswith((".jpg", ".png")):
            name = os.path.splitext(filename)[0]
            filepath = os.path.join(registered_faces_dir, filename)
            image = cv2.imread(filepath, 0)  
            registered_faces[name] = image
    return registered_faces

registered_faces = load_registered_faces()

def recognize_face(detected_face):
    detected_face = cv2.resize(detected_face, (100, 100))  
    detected_face = cv2.equalizeHist(detected_face)  
    registered_faces = load_registered_faces()

    for name, face in registered_faces.items():
        face_resized = cv2.resize(face, (100, 100))
        face_normalized = cv2.equalizeHist(face_resized)

        diff = cv2.absdiff(detected_face, face_normalized)
        score = np.mean(diff)

        if score < 50:  
            return name
    return "Unknown"

def mark_attendance(name, date):
    df = pd.read_csv(attendance_file)

    if date not in df.columns:
        df[date] = "Absent"  

    if name in df["Name"].values:
        df.loc[df["Name"] == name, date] = "Present"
        msg = f"Attendance marked for {name} on {date}.\n"
    elif name not in df["Name"].values:
        new_entry = {"Name": name}
        df = df.append(new_entry, ignore_index=True)
        df.loc[df["Name"] == name, date] = "Present"
        msg = f"Attendance marked for {name} on {date}.\n"
    else:
        msg = f"Error: {name} not found in registered faces."
    df.to_csv(attendance_file, index=False)
    return msg

def calculate_attendance_stats(name):
    if not os.path.exists(attendance_file):
        return
    
    df = pd.read_csv(attendance_file)
    if name not in df["Name"].values:
        msg = f"No attendance data found for {name}."
        return msg
    
    person_data = df.loc[df["Name"] == name]
    total_days = len(df.columns) - 1  
    present_days = (person_data == "Present").sum(axis=1).values[0]
    percentage = (present_days / total_days) * 100 if total_days > 0 else 0
    
    msg = f"Attendance Summary for {name}:\nTotal Days: {total_days}\nDays Present: {present_days}\nAttendance Percentage: {percentage:.2f}%"
    return msg

def add_name_to_attendance(name):
    df = pd.read_csv(attendance_file)
    if name not in df["Name"].values:
        df = pd.concat([df, pd.DataFrame({"Name": [name]})], ignore_index=True)
        df.to_csv(attendance_file, index=False)
        msg = f"Added {name} to the attendance file."
    else:
        msg = f"{name} is already in the attendance file."
    return msg
