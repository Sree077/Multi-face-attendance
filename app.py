from flask import Flask, render_template, Response, redirect, url_for, request, session, flash, make_response
import face_recognition
import cv2
import pickle
import pandas as pd
from datetime import datetime, timedelta
import os
from functools import wraps
from flask_socketio import SocketIO, emit
import json
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure secret key
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)
connected_clients = {}

ENCODINGS_FILE = "encodings.pkl"
ATTENDANCE_FILE = "attendance.csv"
COOLDOWN_PERIOD = timedelta(minutes=30)  # Time before a student can be marked present again

# Course information
COURSES = {
    'CST 468': 'Bioinformatics',
    'CSD 416': 'Project Phase 2',
    'CST 444': 'Soft Computing',
    'CST 466': 'Data Mining',
    'CST 402': 'Distributed Computing',
    'CST 404': 'Comprehensive Course Viva'
}

# Sample user credentials (Replace with database in production)
USERS = {
    # Teachers with their assigned courses
    'teacher1': {
        'password': 'teacher123',
        'role': 'teacher',
        'name': 'Mrs Prameela S ',
        'courses': ['CST 468', 'CSD 416']
    },
    'teacher2': {
        'password': 'teacher456',
        'role': 'teacher',
        'name': 'Mrs Liji Sarah Varghese',
        'courses': ['CST 444']
    },
    'teacher3': {
        'password': 'teacher456',
        'role': 'teacher',
        'name': 'Mrs Rakhimol V',
        'courses': ['CST 466']
    },
    'teacher4': {
        'password': 'teacher456',
        'role': 'teacher',
        'name': 'Mrs Aswathy Priya M',
        'courses': ['CST 402']
    },
    'teacher5': {
        'password': 'teacher456',
        'role': 'teacher',
        'name': 'Mrs Anjana Yesodharan',
        'courses': ['CST 404']
    },
    
    # Students with actual names and sequential passwords
    'Adhira R': {'password': 'password01', 'role': 'student', 'courses': ['CST 468', 'CSD 416', 'CST 444', 'CST 466', 'CST 402', 'CST 404']},
    'Alfiya S': {'password': 'password02', 'role': 'student', 'courses': ['CST 468', 'CSD 416', 'CST 444', 'CST 466', 'CST 402', 'CST 404']},
    'Anjaly C A': {'password': 'password03', 'role': 'student', 'courses': ['CST 468', 'CSD 416', 'CST 444', 'CST 466', 'CST 402', 'CST 404']},
    'Anu Joby': {'password': 'password04', 'role': 'student', 'courses': ['CST 468', 'CSD 416', 'CST 444', 'CST 466', 'CST 402', 'CST 404']},
    'Ashir Muhammad S': {'password': 'password05', 'role': 'student', 'courses': ['CST 468', 'CSD 416', 'CST 444', 'CST 466', 'CST 402', 'CST 404']},
    'Ashkar R': {'password': 'password06', 'role': 'student', 'courses': ['CST 468', 'CSD 416', 'CST 444', 'CST 466', 'CST 402', 'CST 404']},
    'Dany Koshy P': {'password': 'password07', 'role': 'student', 'courses': ['CST 468', 'CSD 416', 'CST 444', 'CST 466', 'CST 402', 'CST 404']},
    'Fathima Hassan': {'password': 'password08', 'role': 'student', 'courses': ['CST 468', 'CSD 416', 'CST 444', 'CST 466', 'CST 402', 'CST 404']},
    'Gopika Krishnan G': {'password': 'password09', 'role': 'student', 'courses': ['CST 468', 'CSD 416', 'CST 444', 'CST 466', 'CST 402', 'CST 404']},
    'Husna Fathima R': {'password': 'password10', 'role': 'student', 'courses': ['CST 468', 'CSD 416', 'CST 444', 'CST 466', 'CST 402', 'CST 404']},
    'Kavita S': {'password': 'password11', 'role': 'student', 'courses': ['CST 468', 'CSD 416', 'CST 444', 'CST 466', 'CST 402', 'CST 404']},
    'Meenu V Nair': {'password': 'password12', 'role': 'student', 'courses': ['CST 468', 'CSD 416', 'CST 444', 'CST 466', 'CST 402', 'CST 404']},
    'Niha Fathima': {'password': 'password13', 'role': 'student', 'courses': ['CST 468', 'CSD 416', 'CST 444', 'CST 466', 'CST 402', 'CST 404']},
    'Nithya S S': {'password': 'password14', 'role': 'student', 'courses': ['CST 468', 'CSD 416', 'CST 444', 'CST 466', 'CST 402', 'CST 404']},
    'Sneha S S': {'password': 'password15', 'role': 'student', 'courses': ['CST 468', 'CSD 416', 'CST 444', 'CST 466', 'CST 402', 'CST 404']},
    'Sona John': {'password': 'password16', 'role': 'student', 'courses': ['CST 468', 'CSD 416', 'CST 444', 'CST 466', 'CST 402', 'CST 404']},
    'Sreelakshmi': {'password': 'password17', 'role': 'student', 'courses': ['CST 468', 'CSD 416', 'CST 444', 'CST 466', 'CST 402', 'CST 404']},
    'Sreelakshmi A': {'password': 'password18', 'role': 'student', 'courses': ['CST 468', 'CSD 416', 'CST 444', 'CST 466', 'CST 402', 'CST 404']},
    'Sumi Mol S': {'password': 'password19', 'role': 'student', 'courses': ['CST 468', 'CSD 416', 'CST 444', 'CST 466', 'CST 402', 'CST 404']},
    'Ujin S Thomas': {'password': 'password20', 'role': 'student', 'courses': ['CST 468', 'CSD 416', 'CST 444', 'CST 466', 'CST 402', 'CST 404']},
}

# Initialize face recognition variables
known_encodings = []
known_names = []
last_seen = {}
camera_running = False
cap = None
current_course_global = None  # Add this global variable

# Load face encodings
try:
    print("Loading face encodings...")
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
        known_encodings = data["encodings"]
        known_names = data["names"]
    print(f"Successfully loaded {len(known_names)} face encodings")
    print("Known names:", known_names)
except Exception as e:
    print(f"Error loading face encodings: {e}")
    known_encodings = []
    known_names = []

# Login decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Role-specific decorators
def teacher_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'role' not in session or session['role'] != 'teacher':
            flash('Access denied. Teacher privileges required.')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def student_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'role' not in session or session['role'] != 'student':
            flash('Access denied. Student privileges required.')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def generate_frames():
    global cap, camera_running, current_course_global
    while camera_running:
        try:
            if not cap or not cap.isOpened():
                print("Camera not opened, attempting to reinitialize...")
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    print("Failed with DirectShow, trying default backend...")
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        print("Failed to open camera with any backend")
                        yield b'--frame\r\n'
                        continue

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                print("Camera initialized successfully")

            success, frame = cap.read()
            if not success or frame is None:
                print("Failed to capture frame")
                yield b'--frame\r\n'
                continue

            if len(known_encodings) == 0:
                print("No face encodings loaded")
                cv2.putText(frame, "No face encodings loaded", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                print(f"Working with {len(known_encodings)} face encodings")

                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    print(f"Error converting frame to RGB: {e}")
                    yield b'--frame\r\n'
                    continue

                face_locations = face_recognition.face_locations(rgb_frame, model="hog", number_of_times_to_upsample=1)
                print(f"Found {len(face_locations)} faces in frame")

                if face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    print(f"Computed encodings for {len(face_encodings)} faces")

                    current_time = datetime.now()

                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
                        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

                        name = "Unknown"
                        if len(face_distances) > 0:
                            best_match_index = face_distances.argmin()
                            if matches[best_match_index]:
                                name = known_names[best_match_index]
                                print(f"Recognized face: {name} (distance: {face_distances[best_match_index]:.2f})")

                                if current_course_global:
                                    # Check if already marked for THIS COURSE today
                                    try:
                                        df = pd.read_csv(ATTENDANCE_FILE) if os.path.exists(
                                            ATTENDANCE_FILE) else pd.DataFrame()
                                        already_marked = False

                                        if not df.empty:
                                            today = current_time.date()
                                            # Check for records with same name, same course, and same day
                                            course_records = df[(df['Name'] == name) &
                                                                (df['Course'] == current_course_global)]
                                            if not course_records.empty:
                                                # Convert Time strings to dates
                                                course_records['Date'] = pd.to_datetime(course_records['Time']).dt.date
                                                # Check if any record exists for today
                                                already_marked = today in course_records['Date'].values

                                        if not already_marked:
                                            last_seen[name] = current_time
                                            try:
                                                df = pd.read_csv(ATTENDANCE_FILE) if os.path.exists(
                                                    ATTENDANCE_FILE) else pd.DataFrame(
                                                    columns=["Name", "Time", "Course", "Status"])
                                                timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                                                new_record = pd.DataFrame(
                                                    [[name, timestamp, current_course_global, "Present"]],
                                                    columns=["Name", "Time", "Course", "Status"])
                                                df = pd.concat([df, new_record], ignore_index=True)
                                                df.to_csv(ATTENDANCE_FILE, index=False)
                                                print(f"Attendance marked for {name} in {current_course_global}")

                                                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                                                cv2.putText(frame,
                                                            f"{name} - Present ({face_distances[best_match_index]:.2f})",
                                                            (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                                            (0, 255, 0), 2)
                                            except Exception as e:
                                                print(f"Error marking attendance: {e}")
                                                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                                                cv2.putText(frame, f"{name} - Error", (left, top - 10),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                        else:
                                            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                                            cv2.putText(frame, f"{name} - Already Marked", (left, top - 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                                    except Exception as e:
                                        print(f"Error checking attendance records: {e}")
                                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                                        cv2.putText(frame, f"{name} - Error", (left, top - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                else:
                                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 165, 0), 2)
                                    cv2.putText(frame, f"{name} - No Course Selected", (left, top - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                            else:
                                print("Unknown face detected")
                                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                                cv2.putText(frame, "Unknown", (left, top - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ret:
                print("Failed to encode frame")
                yield b'--frame\r\n'
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"Error in frame generation: {e}")
            yield b'--frame\r\n'
            continue

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    if session['role'] == 'teacher':
        return redirect(url_for('teacher_dashboard'))
    return redirect(url_for('student_dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']

        if username not in USERS:
            flash('Invalid username. Please check your credentials.')
        elif USERS[username]['password'] != password:
            flash('Incorrect password. Please try again.')
        elif USERS[username]['role'] != role:
            flash('Invalid role selected. Please choose the correct role.')
        else:
            session['username'] = username
            session['role'] = role
            if role == 'teacher':
                return redirect(url_for('teacher_dashboard'))
            else:
                return redirect(url_for('student_dashboard'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/teacher/dashboard')
@login_required
@teacher_required
def teacher_dashboard():
    teacher_courses = USERS[session['username']]['courses']
    course_details = {code: COURSES[code] for code in teacher_courses}
    return render_template('teacher_dashboard.html', 
                         username=session['username'],
                         name=USERS[session['username']]['name'],
                         courses=course_details,
                         camera_running=camera_running,
                         current_course=session.get('current_course', None))

@app.route('/select_course/<course_code>')
@login_required
@teacher_required
def select_course(course_code):
    global current_course_global
    if course_code in USERS[session['username']]['courses']:
        session['current_course'] = course_code
        current_course_global = course_code  # Update the global variable
    return redirect(url_for('teacher_dashboard'))


@app.route('/student/dashboard')
@login_required
@student_required
def student_dashboard():
    try:
        username = session.get('username')
        if not username or username not in USERS:
            session.clear()
            return redirect(url_for('login'))

        # Get student's courses
        student_courses = USERS[username]['courses']

        # Initialize attendance data
        course_attendance = {}

        # Read attendance data
        try:
            df = pd.read_csv(ATTENDANCE_FILE)
        except FileNotFoundError:
            df = pd.DataFrame(columns=['Name', 'Time', 'Course', 'Status'])

        # Set total days to the number of unique dates in the attendance records
        # We'll calculate this per course
        total_days_by_course = {}

        # First pass to calculate total days per course
        for course_code in student_courses:
            # Get all records for this course (from all students)
            course_records = df[df['Course'] == course_code]
            if not course_records.empty:
                # Count unique dates for this course
                unique_dates = course_records['Time'].apply(lambda x: pd.to_datetime(x).date()).unique()
                total_days_by_course[course_code] = len(unique_dates)
            else:
                total_days_by_course[course_code] = 0

        # Process attendance for each course
        for course_code in student_courses:
            # Filter attendance records for this student and course
            course_records = df[(df['Name'] == username) & (df['Course'] == course_code)]

            # Count present days (only where Status is Present or NaN for backward compatibility)
            present_days = 0
            if not course_records.empty:
                # Convert to datetime and get unique dates
                course_records['Date'] = course_records['Time'].apply(lambda x: pd.to_datetime(x).date())
                # Count only dates where status is Present or not marked (for old records)
                present_dates = course_records[
                    (course_records['Status'].isna()) | (course_records['Status'] == 'Present')
                    ]['Date'].unique()
                present_days = len(present_dates)

            # Get total days for this course
            total_days = total_days_by_course.get(course_code, 0)

            # Calculate percentage (avoid division by zero)
            if total_days > 0:
                percentage = round((present_days / total_days) * 100)
            else:
                percentage = 0

            # Determine status
            if percentage >= 75:
                status = 'Good'
            elif percentage >= 50:
                status = 'Warning'
            else:
                status = 'Critical'

            # Get latest attendance record
            latest_attendance = None
            if not course_records.empty:
                latest_record = course_records.iloc[-1]  # Get last record
                latest_time = pd.to_datetime(latest_record['Time'])
                latest_status = latest_record.get('Status', 'Present')  # Default to Present for old records
                latest_attendance = {
                    'date': latest_time.strftime('%Y-%m-%d'),
                    'time': latest_time.strftime('%I:%M %p'),
                    'status': latest_status
                }

            # Store course attendance data
            course_attendance[course_code] = {
                'course_name': COURSES[course_code],
                'present_days': present_days,
                'total_days': total_days,
                'percentage': percentage,
                'status': status,
                'latest_attendance': latest_attendance
            }

        return render_template(
            'student_dashboard.html',
            username=username,
            course_attendance=course_attendance
        )
    except Exception as e:
        print(f"Error in student dashboard: {e}")
        import traceback
        traceback.print_exc()
        flash('An error occurred while loading the dashboard. Please try again.', 'danger')
        return redirect(url_for('login'))
@app.route('/student/attendance/<student_name>')
@login_required
def student_attendance(student_name):
    if session['role'] == 'student' and session['username'] != student_name:
        flash('Access denied')
        return redirect(url_for('student_dashboard'))
    
    df = pd.read_csv(ATTENDANCE_FILE) if os.path.exists(ATTENDANCE_FILE) else pd.DataFrame(columns=["Name", "Time"])
    student_records = df[df['Name'] == student_name]
    return render_template('attendance.html', tables=[student_records.to_html(classes='table table-bordered', index=False)])

@app.route('/start_camera')
@login_required
@teacher_required
def start_camera():
    global cap, camera_running
    if not camera_running:
        try:
            # Release any existing camera instance
            if cap is not None:
                cap.release()
                cap = None
            
            print("Attempting to initialize camera...")
            
            # Try to open the camera with default backend
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Failed to open camera with default backend")
                flash('Error: Could not open camera. Please check if camera is connected.')
                return redirect(url_for('teacher_dashboard'))
            
            print("Camera opened successfully")
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Test if we can read a frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera")
                flash('Error: Could not read from camera. Please check camera permissions.')
                cap.release()
                cap = None
                return redirect(url_for('teacher_dashboard'))
            
            print("Successfully read frame from camera")
            camera_running = True
            flash('Camera started successfully.')
            
        except Exception as e:
            print(f"Error starting camera: {str(e)}")
            flash(f'Error starting camera: {str(e)}')
            if cap is not None:
                cap.release()
                cap = None
            return redirect(url_for('teacher_dashboard'))
    
    return redirect(url_for('camera_view'))

@app.route('/stop_camera')
@login_required
@teacher_required
def stop_camera():
    global cap, camera_running
    if camera_running:
        camera_running = False
        if cap:
            cap.release()
            cap = None
    return redirect(url_for('teacher_dashboard'))

@app.route('/video_feed')
@login_required
@teacher_required
def video_feed():
    if not camera_running:
        return Response(
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            b'\r\n',
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'Access-Control-Allow-Origin': '*',
            'Connection': 'keep-alive'
        }
    )

@app.route('/attendance')
@login_required
def attendance():
    # Initialize DataFrame with required columns if file doesn't exist
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["Name", "Time", "Course"])
    else:
        df = pd.read_csv(ATTENDANCE_FILE)
        # Add Course column if it doesn't exist
        if 'Course' not in df.columns:
            df['Course'] = None
    
    if session['role'] == 'teacher':
        # Filter attendance for teacher's courses
        teacher_courses = USERS[session['username']]['courses']
        if not df.empty:
            df = df[df['Course'].isin(teacher_courses)]
    else:
        # Filter attendance for specific student
        df = df[df['Name'] == session['username']]
    
    # Group by course only if there are records
    grouped_tables = []
    if not df.empty:
        for course in df['Course'].unique():
            if pd.isna(course):  # Skip None/NaN courses
                continue
            course_df = df[df['Course'] == course]
            if not course_df.empty:
                grouped_tables.append({
                    'course_name': COURSES.get(course, 'Unknown Course'),
                    'table': course_df.to_html(classes='table table-bordered', index=False)
                })
    
    # If no grouped tables, show a message
    if not grouped_tables:
        flash('No attendance records found.')
        if session['role'] == 'teacher':
            return redirect(url_for('teacher_dashboard'))
        else:
            return redirect(url_for('student_dashboard'))
    
    return render_template('attendance.html', grouped_tables=grouped_tables)

@app.route('/camera_view')
@login_required
@teacher_required
def camera_view():
    if not camera_running:
        flash('Please start the camera first.')
        return redirect(url_for('teacher_dashboard'))
    
    current_course = session.get('current_course')
    if not current_course:
        flash('Please select a course first.')
        return redirect(url_for('teacher_dashboard'))
    
    return render_template('camera_view.html', 
                         current_course=current_course,
                         course_name=COURSES[current_course])

@app.after_request
def add_security_headers(response):
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self' blob:;"
    return response


@app.route('/test_notification/<username>')
def test_notification(username):
    socketio.emit('attendance_notification',
                  {'message': 'Test notification', 'course': 'TEST', 'time': datetime.now().strftime("%H:%M")},
                  room=username)
    return "Notification sent"


# SocketIO Events
@socketio.on('connect')
def handle_connect():
    print(f"Client connected! SID: {request.sid}")
    if 'username' in session:
        username = session['username']
        connected_clients[username] = request.sid
        print(f"User {username} connected with SID: {request.sid}")
    else:
        print("No username in session during connect")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected! SID: {request.sid}")
    if 'username' in session:
        username = session['username']
        connected_clients.pop(username, None)
        print(f"User {username} disconnected")


# Update the mark_absent route to send notifications
@app.route('/mark_absent', methods=['GET', 'POST'])
@login_required
@teacher_required
def mark_absent():
    if 'current_course' not in session:
        flash('Please select a course first.', 'warning')
        return redirect(url_for('teacher_dashboard'))

    course_code = session['current_course']
    today = datetime.now().date()

    # Get all students enrolled in this course
    all_students = [name for name, data in USERS.items()
                    if data['role'] == 'student' and course_code in data['courses']]

    # Get students who have marked attendance today
    try:
        df = pd.read_csv(ATTENDANCE_FILE) if os.path.exists(ATTENDANCE_FILE) else pd.DataFrame()
        present_students = []
        if not df.empty and 'Course' in df.columns and 'Name' in df.columns and 'Time' in df.columns:
            present_students = df[
                (df['Course'] == course_code) &
                (pd.to_datetime(df['Time']).dt.date == today)
                ]['Name'].unique().tolist()
    except Exception as e:
        print(f"Error reading attendance: {e}")
        flash('Error reading attendance records', 'danger')
        present_students = []

    # Find absent students (enrolled but not present today)
    absent_students = [s for s in all_students if s not in present_students]

    if request.method == 'POST':
        absent_selected = request.form.getlist('absent_students')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            df = pd.read_csv(ATTENDANCE_FILE) if os.path.exists(ATTENDANCE_FILE) else pd.DataFrame(
                columns=["Name", "Time", "Course", "Status"])

            new_records = []
            for student in absent_selected:
                new_records.append({
                    "Name": student,
                    "Time": timestamp,
                    "Course": course_code,
                    "Status": "Absent"
                })

                # Send notification to student if they're connected
                if student in connected_clients:
                    socketio.emit('attendance_notification',
                                  {
                                      'message': f'You have been marked absent for {COURSES[course_code]}',
                                      'course': course_code,
                                      'time': datetime.now().strftime("%H:%M")
                                  },
                                  room=connected_clients[student])

            print(f"Attempting to notify student: {student}")
            print(f"Connected clients: {connected_clients}")
            if student in connected_clients:
                print(f"Sending notification to SID: {connected_clients[student]}")
                socketio.emit('attendance_notification', {
                    'message': f'You have been marked absent for {COURSES[course_code]}',
                    'course': course_code,
                    'time': datetime.now().strftime("%H:%M")
                }, room=connected_clients[student])
            else:
                print(f"Student {student} not currently connected")

            if new_records:
                new_df = pd.DataFrame(new_records)
                df = pd.concat([df, new_df], ignore_index=True)
                df.to_csv(ATTENDANCE_FILE, index=False)
                flash(f'Marked {len(new_records)} students as absent', 'success')
            else:
                flash('No students selected to mark as absent', 'info')

            return redirect(url_for('teacher_dashboard'))

        except Exception as e:
            print(f"Error marking absent: {e}")
            flash('Error marking absent students', 'danger')

    return render_template('mark_absent.html',
                           absent_students=absent_students,
                           course_code=course_code,
                           course_name=COURSES.get(course_code, 'Unknown Course'))


if __name__ == "__main__":
    socketio.run(app, debug=True)
