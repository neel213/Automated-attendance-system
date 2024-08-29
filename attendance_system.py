import csv
import os
import pickle
from datetime import datetime
import cv2
import cvzone
import face_recognition
import numpy as np
from tkinter import simpledialog, messagebox, Tk

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load the background image
imgBackground = cv2.imread('Resources/background.png')

# Load mode images into a list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = [cv2.imread(os.path.join(folderModePath, path)) for path in modePathList]

# Load the encoding file or start fresh
encode_file_path = 'EncodeFile.p'
if os.path.exists(encode_file_path):
    with open(encode_file_path, 'rb') as file:
        encodeListKnownWithIds = pickle.load(file)
    encodeListKnown, studentIds = encodeListKnownWithIds
else:
    encodeListKnown = []
    studentIds = []

print("Encode File Loaded" if os.path.exists(encode_file_path) else "Starting fresh with empty encodings")

# Initialize variables
modeType = 0
counter = 0
id = -1
imgStudent = []
new_person = True

# CSV file setup
csv_file = 'attendance.csv'
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Name", "Major", "Year", "Total Attendance", "Last Attendance Time"])

def ask_for_name():
    root = Tk()
    root.withdraw()
    name = simpledialog.askstring("Input", "Enter your name:")
    root.destroy()
    return name

def save_new_face(name, face_encoding):
    face_images_dir = 'Resources/Faces'
    if not os.path.exists(face_images_dir):
        os.makedirs(face_images_dir)

    face_image_path = os.path.join(face_images_dir, f"{name}.p")
    with open(face_image_path, 'wb') as file:
        pickle.dump(face_encoding, file)

def save_student_photo(name, frame):
    """Save the student's photo in the 'students' directory."""
    students_dir = 'students'
    if not os.path.exists(students_dir):
        os.makedirs(students_dir)

    # Save the image with the student's name
    photo_path = os.path.join(students_dir, f"{name}.jpg")
    cv2.imwrite(photo_path, frame)
    print(f"Photo saved for {name} at {photo_path}")

def mark_attendance(student_id, student_name):
    """Update the attendance CSV file and show notification."""
    updated = False
    studentInfo = {}

    # Load current CSV data
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['ID'] == str(student_id):
                studentInfo = row
                break

    if studentInfo:
        last_attendance_time = datetime.strptime(studentInfo['Last Attendance Time'], "%Y-%m-%d %H:%M:%S")
        seconds_elapsed = (datetime.now() - last_attendance_time).total_seconds()
        if seconds_elapsed > 60:
            studentInfo['Total Attendance'] = str(int(studentInfo.get('Total Attendance', 0)) + 1)
            studentInfo['Last Attendance Time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            updated = True

    if updated:
        # Update the CSV file with new attendance data
        updated_rows = []
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['ID'] == str(student_id):
                    row['Total Attendance'] = studentInfo['Total Attendance']
                    row['Last Attendance Time'] = studentInfo['Last Attendance Time']
                updated_rows.append(row)

        with open(csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=updated_rows[0].keys())
            writer.writeheader()
            writer.writerows(updated_rows)

        # Display a pop-up notification
        root = Tk()
        root.withdraw()
        messagebox.showinfo("Attendance Marked", f"Attendance marked for {student_name}")
        root.destroy()

        # Show black screen for 2 seconds
        imgBlack = np.zeros_like(imgBackground)
        cv2.imshow("Face Attendance", imgBlack)
        cv2.waitKey(2000)  # Wait for 2 seconds

        return True
    else:
        return False

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    try:
        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
    except Exception as e:
        print(f"Error during face detection/encoding: {e}")
        continue

    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    face_detected = False

    if faceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            if encodeListKnown:  # Check if there are any known encodings
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                    imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                    id = studentIds[matchIndex]
                    student_name = studentIds[matchIndex]
                    face_detected = True

                    if mark_attendance(id, student_name):
                        save_student_photo(student_name, img)
                    new_person = False

            else:
                if new_person:
                    name = ask_for_name()
                    if name:
                        save_new_face(name, encodeFace)
                        studentIds.append(name)  # Append new name
                        encodeListKnown.append(encodeFace)

                        # Save new faces to the EncodeFile
                        with open(encode_file_path, 'wb') as file:
                            pickle.dump((encodeListKnown, studentIds), file)

                        save_student_photo(name, img)
                        mark_attendance(len(studentIds) - 1, name)

                        new_person = False  # Reset the flag for new person

        if counter != 0:
            if counter == 1 and not new_person:
                # Display student information
                studentInfo = {}

                with open(csv_file, mode='r') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        if row['ID'] == str(id):
                            studentInfo = row
                            break

                if studentInfo:
                    datetimeObject = datetime.strptime(
                        studentInfo.get('Last Attendance Time', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        "%Y-%m-%d %H:%M:%S")
                    secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                    if secondsElapsed > 30:
                        studentInfo['Total Attendance'] = str(int(studentInfo.get('Total Attendance', 0)) + 1)
                        studentInfo['Last Attendance Time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        updated = True

                if updated:
                    # Update the CSV file with new attendance data
                    updated_rows = []
                    with open(csv_file, mode='r') as file:
                        reader = csv.DictReader(file)
                        for row in reader:
                            if row['ID'] == str(id):
                                row['Total Attendance'] = studentInfo['Total Attendance']
                                row['Last Attendance Time'] = studentInfo['Last Attendance Time']
                            updated_rows.append(row)

                    with open(csv_file, mode='w', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=updated_rows[0].keys())
                        writer.writeheader()
                        writer.writerows(updated_rows)

                    # Display a pop-up notification
                    root = Tk()
                    root.withdraw()
                    messagebox.showinfo("Attendance Marked", f"Attendance marked for {studentInfo['Name']}")
                    root.destroy()

                    # Show black screen for 2 seconds
                    imgBlack = np.zeros_like(imgBackground)
                    cv2.imshow("Face Attendance", imgBlack)
                    cv2.waitKey(2000)  # Wait for 2 seconds

                    # Resume normal display
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
                    counter = 0
                    new_person = True  # Reset the flag for new person

            if modeType != 3:
                if 10 < counter < 20:
                    modeType = 2

                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                if counter <= 10:
                    cv2.putText(imgBackground, str(studentInfo.get('Total Attendance', '')), (861, 125),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentInfo.get('Major', '')), (1006, 550),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(id), (1006, 493),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentInfo.get('Year', '')), (1025, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                    (w, h), _ = cv2.getTextSize(studentInfo.get('Name', ''), cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    offset = (414 - w) // 2
                    cv2.putText(imgBackground, str(studentInfo.get('Name', '')), (808 + offset, 445),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                counter += 1

                if counter >= 20:
                    counter = 0
                    modeType = 0
                    studentInfo = {}
                    imgStudent = []
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    else:
        modeType = 0
        counter = 0

    cv2.imshow("Face Attendance", imgBackground)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
