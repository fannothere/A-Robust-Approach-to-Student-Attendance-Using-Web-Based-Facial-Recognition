import cv2
import numpy as np
from ultralytics import YOLO
import mysql.connector
from mysql.connector import pooling
from keras.models import load_model
from datetime import datetime
from keras.applications.mobilenet_v2 import preprocess_input
from flask import Flask, Response, render_template, request, jsonify
import json
import time

app = Flask(__name__)

# Database configuration
db_config = {
    "pool_name": "mypool",
    "pool_size": 5,
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "attendance_db"
}

try:
    connection_pool = mysql.connector.pooling.MySQLConnectionPool(**db_config)
except mysql.connector.Error as err:
    print(f"Error creating connection pool: {err}")
    exit(1)

# Load models and label maps
face_detector = YOLO('yolov8n-face.pt')  # Use YOLO for face detection
face_recognizer = load_model('face_model.h5')  # Load pre-trained face recognition model
label_map = np.load('label_face.npy', allow_pickle=True).item()  # Load label map
label_map_rev = {v: k for k, v in label_map.items()}  # Reverse the label map for lookup

def get_db_connection():
    """Get a connection from the pool with retry logic"""
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            connection = connection_pool.get_connection()
            return connection
        except mysql.connector.Error as err:
            if attempt == max_retries - 1:
                raise err
            time.sleep(retry_delay)
            continue

def preprocess_image(img):
    """Preprocess image for face recognition"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 160))
    img = preprocess_input(img)
    return img

def recognize_face(face_img):
    """Recognize a face in the given image"""
    face = preprocess_image(face_img)
    face = np.expand_dims(face, axis=0)
    prediction = face_recognizer.predict(face)
    accuracy = np.max(prediction)  
    label = np.argmax(prediction)
    return label, accuracy

def safe_execute_query(cursor, query, params=None):
    """Execute a query with retry logic"""
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return True
        except mysql.connector.Error as err:
            if attempt == max_retries - 1:
                print(f"Query failed after {max_retries} attempts: {err}")
                return False
            time.sleep(retry_delay)
            continue

def gen_frames(video_source='file'):
    """Generate frames from video source (either file or webcam)"""
    # If video source is 'file', use the provided video file
    if video_source == 'file':
        cap = cv2.VideoCapture('Student.MOV')
    # If video source is 'webcam', use the webcam
    elif video_source == 'webcam':
        cap = cv2.VideoCapture(0)
    else:
        raise ValueError("Invalid video source")

    connection = None
    cursor = None
    
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces using YOLO
            results = face_detector(frame)
            total_faces = 0
            recognized_faces = 0
            unrecognized_faces = 0
            
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    total_faces += 1
                    
                    face_img = frame[max(0, y1):min(frame.shape[0], y2), 
                                   max(0, x1):min(frame.shape[1], x2)]
                    
                    if face_img.size == 0 or face_img.shape[0] == 0 or face_img.shape[1] == 0:
                        continue
                    
                    try:
                        label, accuracy = recognize_face(face_img)
                        if accuracy > 0.7:
                            recognized_faces += 1
                            class_name = label_map_rev[label]
                            name, nim = class_name.split('_', 1)

                            # Insert attendance record into the database
                            sql_insert = """
                                INSERT INTO attendance (name, nim, accuracy, attendance_time) 
                                VALUES (%s, %s, %s, %s)
                            """
                            val_attendance = (name, nim, float(accuracy), datetime.now())
                            
                            if safe_execute_query(cursor, sql_insert, val_attendance):
                                connection.commit()
                            
                            cv2.putText(frame, f"{name} ({accuracy:.2f})", 
                                      (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, (0, 255, 0), 2)
                        else:
                            unrecognized_faces += 1
                            cv2.putText(frame, "Tidak Dikenali", 
                                      (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, (0, 0, 255), 2)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    except Exception as e:
                        print(f"Error processing face: {e}")
                        continue
            
            # Update stats in the database
            sql_stats = """
                UPDATE face_stats 
                SET total_faces=%s, recognized_faces=%s, unrecognized_faces=%s 
                WHERE id=1
            """
            val_stats = (total_faces, recognized_faces, unrecognized_faces)
            
            if safe_execute_query(cursor, sql_stats, val_stats):
                connection.commit()
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    except Exception as e:
        print(f"Error in gen_frames: {e}")
    
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        if cap:
            cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    source = request.args.get('source', 'file')  # Default to 'file' if no parameter is provided
    
    if source == 'file':
        return Response(gen_frames(video_source='file'),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    elif source == 'webcam':
        return Response(gen_frames(video_source='webcam'),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Invalid video source", 400

@app.route('/get_stats')
def get_stats():
    """Get face detection statistics"""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        cursor.execute("SELECT * FROM face_stats WHERE id=1")
        stats = cursor.fetchone()
        
        return jsonify({
            'total_faces': stats[1],
            'recognized_faces': stats[2],
            'unrecognized_faces': stats[3]
        })
    
    except Exception as e:
        print(f"Error getting stats: {e}")
        return jsonify({
            'total_faces': 0,
            'recognized_faces': 0,
            'unrecognized_faces': 0
        })
    
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

@app.route('/get_attendance')
def get_attendance():
    """Get attendance records within a time range"""
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        start_datetime = request.args.get('start_datetime', 
                                          datetime.now().strftime('%Y-%m-%d 00:00:00'))
        end_datetime = request.args.get('end_datetime', 
                                        datetime.now().strftime('%Y-%m-%d 23:59:59'))
        sql = """
            SELECT name, nim, MAX(accuracy) AS highest_accuracy, MAX(attendance_time) AS last_time
            FROM attendance
            WHERE attendance_time BETWEEN %s AND %s
            GROUP BY name, nim
            ORDER BY last_time DESC
        """
        cursor.execute(sql, (start_datetime, end_datetime))
        attendance_data = cursor.fetchall()
        return jsonify([{
            'name': row[0],
            'nim': row[1],
            'accuracy': round(row[2] * 100, 1), 
            'timestamp': row[3].strftime('%Y-%m-%d %H:%M:%S')
        } for row in attendance_data])
    except Exception as e:
        print(f"Error getting attendance: {e}")
        return jsonify([])

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

if __name__ == '__main__':
    app.run(debug=True)
