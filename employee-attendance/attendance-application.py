import sqlite3
from datetime import datetime, timedelta
import sys
import os
import torch
import cv2
import time
import queue
import threading
import logging
import openpyxl
import pandas as pd
from PyQt5.QtWidgets import (QPushButton, QMessageBox, QLineEdit, QDateEdit, QGroupBox, QHBoxLayout,
                             QHeaderView, QApplication, QSplitter, QVBoxLayout, QLabel, QTableWidget,
                             QTableWidgetItem, QWidget, QHBoxLayout, QFrame, QFileDialog)
from PyQt5.QtCore import QTimer, Qt, QPoint, QDate
from PyQt5.QtGui import QImage, QPixmap, QBrush, QColor
from torchvision import transforms as trans
import argparse
from PIL import Image, ImageDraw, ImageFont
from utils import *
from utils.align_trans import *
from MTCNN.MTCNN import create_mtcnn_net
from face_model import MobileFaceNet, l2_norm
from facebank import load_facebank, prepare_facebank
import numpy as np


# Set up logging
logging.basicConfig(filename="AttendanceApplication_Errorlogs.log", filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# Thread-local storage for SQLite connection to avoid conflicts in a multithreaded environment
thread_local = threading.local()

def resource_path(relative_path):
    """Get absolute path to resource for development and PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def resource_path_facebank(relative_path):
    base_path = os.path.dirname(os.path.abspath(sys.argv[0]))  # Directory of the .exe file
    return os.path.join(base_path, relative_path)

def get_db_connection():
    """Create or retrieve a SQLite connection from thread-local storage."""
    if not hasattr(thread_local, 'db_connection'):
        thread_local.db_connection = sqlite3.connect('attendance_data.db', check_same_thread=False)
    return thread_local.db_connection

def close_db_connection():
    """Close SQLite connection stored in thread-local storage if it exists."""
    if hasattr(thread_local, 'db_connection'):
        thread_local.db_connection.close()
        del thread_local.db_connection


# Initialize SQLite database and create tables if they don't exist
conn = get_db_connection()
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, date DATE, first_login_time TIME,
    last_logout_time TIME, current_status TEXT, total_work_hours TEXT DEFAULT '00:00:00',
    total_break_hours TEXT DEFAULT '00:00:00', UNIQUE(name, date))''')
cursor.execute('''CREATE TABLE IF NOT EXISTS attendance_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, attendance_id INTEGER, login_time TIME, 
    logout_time TIME, FOREIGN KEY(attendance_id) REFERENCES attendance(id))''')

def log_attendance(name):
    """Logs attendance and manages sessions and statuses based on the person's name."""
    today = datetime.now().date()
    
    # Retrieve today's attendance record for the given name
    cursor.execute('''SELECT id, current_status FROM attendance WHERE name = ? AND date = ?''', (name, today))
    record = cursor.fetchone()

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Current timestamp

    if record is None:
        # Log initial entry if no attendance record exists for today
        cursor.execute('''INSERT INTO attendance (name, date, first_login_time, current_status)
                          VALUES (?, ?, ?, 'Logged In')''', (name, today, current_time))
        attendance_id = cursor.lastrowid

        # Log a new session with login_time
        cursor.execute('''INSERT INTO attendance_sessions (attendance_id, login_time)
                          VALUES (?, ?)''', (attendance_id, current_time))
    else:
        attendance_id, current_status = record

        # Check for an ongoing session without a logout time
        cursor.execute('''SELECT id, login_time FROM attendance_sessions 
                          WHERE attendance_id = ? AND logout_time IS NULL
                          ORDER BY login_time DESC LIMIT 1''', (attendance_id,))
        ongoing_session = cursor.fetchone()

        if ongoing_session:
            # Update logout time if a significant period has passed
            session_id, login_time = ongoing_session

            # Update logout time only if significant time has passed (e.g., more than 10 seconds)
            if datetime.now() - datetime.fromisoformat(login_time) > timedelta(seconds=10):
                cursor.execute('''UPDATE attendance_sessions SET logout_time = ? WHERE id = ?''', (current_time, session_id))
                cursor.execute('''UPDATE attendance SET current_status = 'Logged Out', last_logout_time = ? WHERE id = ?''', (current_time, attendance_id))
        else:
            # No ongoing session, check if the user was logged out recently
            # Handle session continuation and break handling
            cursor.execute('''SELECT login_time, logout_time FROM attendance_sessions 
                              WHERE attendance_id = ? ORDER BY login_time DESC LIMIT 1''', (attendance_id,))
            last_session = cursor.fetchone()

            if last_session:  # Ensure last session has a logout time
                last_login_time, last_logout_time = last_session
                
                if last_logout_time:  # Check if the last session has a logout time
                    # Calculate the break duration
                    break_duration = datetime.now() - datetime.fromisoformat(last_logout_time)
                    if break_duration > timedelta(minutes=1):  # Allow a break time if significant
                        # Create a new session if the last session was logged out properly
                        cursor.execute('''INSERT INTO attendance_sessions (attendance_id, login_time)
                                          VALUES (?, ?)''', (attendance_id, current_time))
                        
                        # Update attendance status to 'Logged In'
                        cursor.execute('''UPDATE attendance SET current_status = 'Logged In' WHERE id = ?''', (attendance_id,))


                else:
                    # If the last session does not have a logout time, update that session instead or Finalize session if previous logout time is missing
                    cursor.execute('''UPDATE attendance_sessions SET logout_time = ? WHERE id = ?''', (current_time, last_session[0]))
                    cursor.execute('''UPDATE attendance SET current_status = 'Logged Out', last_logout_time = ? WHERE id = ?''', (current_time, attendance_id))

    conn.commit()
    calculate_hours(name)

def calculate_hours(name):
    """Calculate and update total work and break hours for the given name."""
    
    today = datetime.now().date()

    # Fetch the attendance record for today
    cursor.execute('SELECT id FROM attendance WHERE name = ? AND date = ?', (name, today))
    attendance_record = cursor.fetchone()

    if attendance_record:
        attendance_id = attendance_record[0]

        # Fetch all sessions for the user for today
        cursor.execute('''
        SELECT login_time, logout_time FROM attendance_sessions
        WHERE attendance_id = ?
        ORDER BY login_time ASC
        ''', (attendance_id,))
        
        sessions = cursor.fetchall()

        # Initialize total work and break time
        total_work_time = timedelta()
        total_break_time = timedelta()
        previous_logout_time = None

        # Calculate work and break hours
        for login_time, logout_time in sessions:
            # Convert the strings to datetime objects
            login_time = datetime.fromisoformat(login_time)

            if logout_time:
                logout_time = datetime.fromisoformat(logout_time)
                session_duration = logout_time - login_time
                total_work_time += session_duration
            else:
                # Session is ongoing, calculate till now
                session_duration = datetime.now() - login_time
                total_work_time += session_duration
            
            if previous_logout_time:
                # Calculate break time between sessions
                break_duration = login_time - previous_logout_time
                total_break_time += break_duration

            # Update previous logout time
            previous_logout_time = logout_time if logout_time else datetime.now()

        # Format the time as HH:MM:SS
        total_work_str = str(total_work_time)
        total_work_str = total_work_str.split('.')[0]
        total_break_str = str(total_break_time)

        if total_work_time > timedelta(hours=8):
            # Update attendance status to 'Logged In'
            cursor.execute('''UPDATE attendance SET current_status = 'Overtime' WHERE id = ?''', (attendance_id,))


        # Update the attendance table with total work and break hours
        cursor.execute('''
        UPDATE attendance
        SET total_work_hours = ?, total_break_hours = ?
        WHERE id = ?
        ''', (total_work_str, total_break_str, attendance_id))
        
        conn.commit()

        return total_work_str, total_break_str
    else:
        return None, None

# Threading for video frame processing
frame_queue = queue.Queue(maxsize=4)
result_queue = queue.Queue(maxsize=4)
stop_threads = False

def resize_image(img, scale):
    """Resize an image by the specified scale."""
    height, width, channel = img.shape
    new_height = int(height * scale)
    new_width = int(width * scale)
    img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return img_resized

def capture_frames(cap):
    """Capture frames from video source, resizing and storing in the frame queue."""
    global stop_threads
    while not stop_threads:
        ret, frame = cap.read()
        if not ret or stop_threads:
            break
        # Resized the image for 2K resolution camera to 720p
        frame_resized = resize_image(frame, 0.5) if frame.shape[1] == 2560 else frame
        if not frame_queue.full():
            frame_queue.put(frame_resized)
        else:
            try:
                frame_queue.get_nowait()
                frame_queue.put(frame_resized)
            except queue.Empty:
                pass

def process_frames(args, device, detect_model, targets, names):
    """Process frames for face detection and recognition, logging attendance if matched."""
    prev_bboxes, prev_landmarks, prev_results, prev_score_100 = [], [], [], []
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            try:
                input = resize_image(frame, args.scale)
                bboxes, landmarks = create_mtcnn_net(
                    input, args.mini_face, device, 
                    p_model_path=resource_path('MTCNN/weights/pnet_Weights'),
                    r_model_path=resource_path('MTCNN/weights/rnet_Weights'),
                    o_model_path=resource_path('MTCNN/weights/onet_Weights')
                )

                if stop_threads:
                    break

                if bboxes is not None and len(bboxes) > 0:
                    bboxes = bboxes / args.scale
                    landmarks = landmarks / args.scale

                    faces = Face_alignment(frame, default_square=True, landmarks=landmarks)
                    embs = []
                    test_transform = trans.Compose([
                        trans.ToTensor(),
                        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ])

                    for img in faces:
                        emb = detect_model(test_transform(img).to(device).unsqueeze(0))
                        embs.append(emb)

                    if embs:
                        source_embs = torch.cat(embs)
                        diff = source_embs.unsqueeze(-1) - targets.transpose(1, 0).unsqueeze(0)
                        dist = torch.sum(torch.pow(diff, 2), dim=1)
                        minimum, min_idx = torch.min(dist, dim=1)
                        min_idx[minimum > ((args.threshold-156)/(-80))] = -1
                        results = min_idx
                        score_100 = torch.clamp(minimum * -80 + 156, 0, 100)

                        for i, result in enumerate(results):
                            if result != -1:
                                recognized_name = names[result + 1]
                                log_attendance(recognized_name)

                else:
                    # No faces detected
                    results = []
                    score_100 = []
                    bboxes = []

                prev_bboxes, prev_landmarks, prev_results, prev_score_100 = bboxes, landmarks, results, score_100
                result_queue.put((frame, prev_bboxes, prev_landmarks, prev_results, prev_score_100))

            except Exception as e:
                logging.error('Detection error:%s', e)
                result_queue.put((frame, [], [], [], []))  # Ensure queue is updated in case of error
                if stop_threads:
                    break

class AttendanceUI(QWidget):
    def __init__(self, names, capture_thread, process_thread, cap):
        super().__init__()
        self.names = names
        self.setWindowTitle("Real-Time Attendance & Face Recognition")
        self.capture_thread = capture_thread
        self.process_thread = process_thread
        self.cap = cap

        # Window and Styling Settings
        # Set frameless window and translucent background
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Load external stylesheet
        with open(resource_path('utils/Obit.qss'), "r") as f:
            
            style = f.read()
            self.setStyleSheet(style)

        # Create a frame for rounded corners
        frame = QFrame(self)
        frame.setStyleSheet("border-radius: 20px; background-color: rgba(27 ,38 ,59, 0.9);")
        frame.setContentsMargins(10, 10, 10, 10) 

        # Create main layout
        main_layout = QVBoxLayout(frame)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

         # Title Bar Layout with Title, Minimize, and Close Buttons
        title_bar = QWidget()
        title_bar_layout = QHBoxLayout()
        title_bar_layout.setContentsMargins(0, 0, 0, 0)

        title_label = QLabel("Employee Face Attendance System")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 20px; font-weight: bold;")

        # Minimize button
        minimize_button = QPushButton("-")
        minimize_button.setFixedSize(30, 30)
        minimize_button.setStyleSheet("QPushButton {background-color: lightblue; color: white; border-radius: 5px;} QPushButton:hover {background-color: darkblue;}")
        minimize_button.clicked.connect(self.showMinimized)

        # Close button
        close_button = QPushButton("x")
        close_button.setFixedSize(30, 30)
        close_button.setStyleSheet("QPushButton {background-color: red; color: white; border-radius: 5px;} QPushButton:hover {background-color: darkred;}")
        close_button.clicked.connect(self.close)

        title_bar_layout.addWidget(title_label)
        title_bar_layout.addWidget(minimize_button)
        title_bar_layout.addWidget(close_button)
        title_bar.setLayout(title_bar_layout)
        main_layout.addWidget(title_bar)

        # Filter Layout with Search and Date Filter
        filter_group = QGroupBox()
        filter_layout = QHBoxLayout()

        # Search Input
        self.search_line_edit = QLineEdit(self)
        self.search_line_edit.setPlaceholderText("Search by name")
        self.search_line_edit.setStyleSheet("border-radius: 5px;")
        self.search_line_edit.textChanged.connect(self.filter_attendance_data)

        # Date Filter Input
        self.date_label = QLabel("Filter by Date:")
        self.date_edit = QDateEdit(self)
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.setStyleSheet("color: #e6e8e6;")
        self.date_edit.setDisplayFormat("yyyy-MM-dd")  # Set the date format
        self.date_edit.dateChanged.connect(self.filter_attendance_data)  # Connect date change to filtering

        filter_layout.addWidget(self.search_line_edit)
        filter_layout.addWidget(self.date_label)
        filter_layout.addWidget(self.date_edit)
        filter_group.setLayout(filter_layout)
        main_layout.addWidget(filter_group)

        # Splitter for Video Feed and Attendance Table
        splitter = QSplitter(Qt.Vertical)

        # Video Display
        self.face_recognition_label = QLabel()
        self.face_recognition_label.setMinimumSize(400, 300)
        self.face_recognition_label.setStyleSheet("border: 1px solid black;")
        splitter.addWidget(self.face_recognition_label)

        # Attendance Table
        self.attendance_table = QTableWidget()
        self.attendance_table.verticalHeader().setVisible(False)
        self.attendance_table.setColumnCount(8)
        self.attendance_table.setHorizontalHeaderLabels(["ID", "Name", "Date", "Login Time", "Log-Out Time", "Status", "Working Hours", "Total Break Time"])
        self.attendance_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)  # Automatically adjust column widths
        self.attendance_table.horizontalHeader().setStyleSheet("font-weight: bold; font-size:16px")
        self.attendance_table.cellClicked.connect(self.on_cell_clicked)
        splitter.addWidget(self.attendance_table)
        splitter.setSizes([700, 300])
        main_layout.addWidget(splitter)

        # Button layout
        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 10, 0, 0)  
        buttons_layout.setSpacing(10)

        # Back button
        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(self.show_full_attendance)
        self.back_button.setVisible(False)  # Hide initially
        self.back_button.setFixedSize(155, 40)
        self.back_button.setStyleSheet("""QPushButton {background-color: #4CAF50; color: white; font-size: 16px; border-radius: 8px;} QPushButton:hover {background-color: #45a049;} QPushButton:pressed {background-color: #3e8e41;}""")

        # Download report button
        self.download_button = QPushButton("Download Report")
        self.download_button.clicked.connect(self.download_report)
        self.download_button.setFixedSize(155, 40)
        self.download_button.setStyleSheet("""QPushButton {background-color: #FF9800; color: white; font-size: 16px; border-radius: 8px;} QPushButton:hover {background-color: #FB8C00;} QPushButton:pressed {background-color: #F57C00;}""")

        buttons_layout.addWidget(self.download_button)
        buttons_layout.addWidget(self.back_button)


        # Date Filter Toggle Button
        self.date_filter_button = QPushButton("Date Filter")
        self.date_filter_button.setFixedSize(130, 30)
        self.date_filter_button.setStyleSheet("""QPushButton {background-color: #bc4749; color: white; border-radius: 5px;}
                                                QPushButton:hover {background-color: #e63946;}""")
        self.date_filter_button.clicked.connect(self.toggle_date_filter)
        filter_layout.addWidget(self.date_filter_button)

        # Initially hide the date filter elements
        self.date_edit.setVisible(False)
        self.date_label.setVisible(False)
        
        # Finalize Layout
        main_layout.addLayout(buttons_layout)

        # Set layout to frame
        frame.setLayout(main_layout)

        # Set the layout for the main window
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(frame)

        # Timer for updating the attendance table and video feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_attendance_table)
        self.timer.start(1000)

        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_video_feed)
        self.video_timer.start(30)

        self.load_attendance_data()  # Load all attendance data initially

    # Define the method to clear the date filter
    def clear_date_filter(self):
        self.date_edit.setDate(QDate.currentDate())  # Reset the date to current (or any default date)
        self.load_attendance_data()  # Reload the full data without filters

    # Toggle Date Filter Visibility
    def toggle_date_filter(self):
        if self.date_edit.isVisible():
            # Hide the date filter
            self.date_edit.setVisible(False)
            self.date_label.setVisible(False)
            self.date_filter_button.setText("Date Filter")
            self.load_attendance_data()  # Reload full data when date filter is cleared
        else:
            # Show the date filter
            self.date_edit.setVisible(True)
            self.date_label.setVisible(True)
            self.date_filter_button.setText("Clear Filter")

            # Set the date to today automatically
            today_date = QDate.currentDate()
            self.date_edit.setDate(today_date)

            # Immediately filter attendance data for today's date
            self.filter_attendance_data()


    def load_attendance_data(self, name=None, date=None):
        self.search_line_edit.clear()  # Clear search line edit when loading data
        self.date_edit.clear()
        cursor = conn.cursor()
        self.attendance_table.clearContents()  # Clear table contents to avoid residual data
        
        if name and date:
            self.displaying_sessions = True  # Set flag for session view
            
            # Query attendance_sessions for specific user
            cursor.execute('''SELECT attendance_id, name, login_time, logout_time FROM attendance_sessions 
                            JOIN attendance ON attendance.id = attendance_sessions.attendance_id
                            WHERE attendance.name = ? and attendance.date = ?''', (name,date))
            records = cursor.fetchall()
            
            # Set up table for session data
            self.attendance_table.setRowCount(len(records))
            self.attendance_table.setColumnCount(4)
            self.attendance_table.setHorizontalHeaderLabels(["Attendance ID", "Name", "Login Time", "Logout Time"])
            
            for row_idx, (attendance_id, name, login_time, logout_time) in enumerate(records):
                self.attendance_table.setItem(row_idx, 0, QTableWidgetItem(str(attendance_id)))
                self.attendance_table.setItem(row_idx, 1, QTableWidgetItem(str(name)))
                self.attendance_table.setItem(row_idx, 2, QTableWidgetItem(str(login_time)))
                self.attendance_table.setItem(row_idx, 3, QTableWidgetItem(str(logout_time)))

                # Set alignment and padding using stylesheets
                for col_idx in range(4):
                    item = self.attendance_table.item(row_idx, col_idx)
                    item.setTextAlignment(Qt.AlignCenter)  
                    item.setBackground(QBrush(QColor(29, 53, 87)))  

            self.back_button.setVisible(True)  # Show back button
        else:
            self.displaying_sessions = False  # Reset flag for general view

            # Query and load main attendance data
            cursor.execute('''SELECT id, name, date, strftime('%H:%M:%S', first_login_time) AS first_login_time, 
                            strftime('%H:%M:%S', last_logout_time) AS last_logout_time, current_status, 
                            total_work_hours, total_break_hours FROM attendance''')
            records = cursor.fetchall()

            self.filter_attendance_data() 
            
            
            # Set up table for main attendance data
            self.attendance_table.setRowCount(len(records))
            self.attendance_table.setColumnCount(8)
            self.attendance_table.setHorizontalHeaderLabels(["ID", "Name", "Date", "Login Time", "Log-Out Time", "Status", "Working Hours", "Total Break Time"])
            
            for row_idx, (id, name, date, first_login, last_logout, status, work_hours, break_hours) in enumerate(records):
                self.attendance_table.setItem(row_idx, 0, QTableWidgetItem(str(id)))
                self.attendance_table.setItem(row_idx, 1, QTableWidgetItem(name))
                self.attendance_table.setItem(row_idx, 2, QTableWidgetItem(str(date)))
                self.attendance_table.setItem(row_idx, 3, QTableWidgetItem(str(first_login)))
                self.attendance_table.setItem(row_idx, 4, QTableWidgetItem(str(last_logout)))
                self.attendance_table.setItem(row_idx, 5, QTableWidgetItem(status))
                self.attendance_table.setItem(row_idx, 6, QTableWidgetItem(str(work_hours)))
                self.attendance_table.setItem(row_idx, 7, QTableWidgetItem(str(break_hours)))

                # Set alignment and padding using stylesheets
                for col_idx in range(8):
                    item = self.attendance_table.item(row_idx, col_idx)
                    item.setTextAlignment(Qt.AlignCenter)  
                    item.setBackground(QBrush(QColor(29, 53, 87)))  

            self.attendance_table.horizontalHeader().setStyleSheet("font-weight: bold; font-size:16px;")

            self.back_button.setVisible(False)  

        # Start or restart the timer only if displaying general attendance data
        if not name:  
            self.timer.timeout.disconnect()  
            self.timer.timeout.connect(self.update_attendance_data)  
            self.timer.start(1000)  # Update every second (adjust as necessary)
        else:
            self.timer.stop()  # Stop the timer when showing specific user sessions


    def show_all_data(self):
        """Clear filters and display all attendance data in the table."""
        self.search_line_edit.clear()  
        self.date_edit.setDate(QDate.currentDate())   
        

        # Load all data without filtering
        cursor = conn.cursor()
        cursor.execute('''SELECT id, name, date, strftime('%H:%M:%S', first_login_time) AS first_login_time, 
                        strftime('%H:%M:%S', last_logout_time) AS last_logout_time, current_status, 
                        total_work_hours, total_break_hours FROM attendance''')
        records = cursor.fetchall()

        # Populate table with all records
        self.attendance_table.setRowCount(len(records))
        for row_idx, (id, name, date, first_login, last_logout, status, work_hours, break_hours) in enumerate(records):
            self.attendance_table.setItem(row_idx, 0, QTableWidgetItem(str(id)))
            self.attendance_table.setItem(row_idx, 1, QTableWidgetItem(name))
            self.attendance_table.setItem(row_idx, 2, QTableWidgetItem(str(date)))
            self.attendance_table.setItem(row_idx, 3, QTableWidgetItem(str(first_login)))
            self.attendance_table.setItem(row_idx, 4, QTableWidgetItem(str(last_logout)))
            self.attendance_table.setItem(row_idx, 5, QTableWidgetItem(status))
            self.attendance_table.setItem(row_idx, 6, QTableWidgetItem(str(work_hours)))
            self.attendance_table.setItem(row_idx, 7, QTableWidgetItem(str(break_hours)))


            for col_idx in range(8):
                item = self.attendance_table.item(row_idx, col_idx)
                item.setTextAlignment(Qt.AlignCenter)
                item.setBackground(QBrush(QColor(29, 53, 87)))

        self.attendance_table.horizontalHeader().setStyleSheet("font-weight: bold; font-size:16px;")

    def filter_attendance_data(self):
        # Get the text from the search line edit
        search_text = self.search_line_edit.text().strip().lower()
        
        # Check if the date filter is active
        date_filter_active = self.date_edit.isVisible()
        selected_date = self.date_edit.date().toString("yyyy-MM-dd") if date_filter_active else None

        cursor = conn.cursor()

        # Construct the query based on active filters
        if search_text and selected_date:  # Both name and date are filtered
            query = '''SELECT id, name, date, strftime('%H:%M:%S', first_login_time) AS first_login_time,
                       strftime('%H:%M:%S', last_logout_time) AS last_logout_time, current_status,
                       total_work_hours, total_break_hours FROM attendance
                       WHERE LOWER(name) LIKE ? AND date = ?'''
            cursor.execute(query, (f"%{search_text}%", selected_date))
        elif search_text:  # Only name is filtered
            query = '''SELECT id, name, date, strftime('%H:%M:%S', first_login_time) AS first_login_time,
                       strftime('%H:%M:%S', last_logout_time) AS last_logout_time, current_status,
                       total_work_hours, total_break_hours FROM attendance
                       WHERE LOWER(name) LIKE ?'''
            cursor.execute(query, (f"%{search_text}%",))
        elif selected_date:  # Only date is filtered
            query = '''SELECT id, name, date, strftime('%H:%M:%S', first_login_time) AS first_login_time,
                       strftime('%H:%M:%S', last_logout_time) AS last_logout_time, current_status,
                       total_work_hours, total_break_hours FROM attendance
                       WHERE date = ?'''
            cursor.execute(query, (selected_date,))
        else:  # No filter
            query = '''SELECT id, name, date, strftime('%H:%M:%S', first_login_time) AS first_login_time,
                       strftime('%H:%M:%S', last_logout_time) AS last_logout_time, current_status,
                       total_work_hours, total_break_hours FROM attendance'''
            cursor.execute(query)
        
        records = cursor.fetchall()
        
        # Update table with filtered data
        self.attendance_table.setRowCount(len(records))
        for row_idx, (id, name, date, first_login, last_logout, status, work_hours, break_hours) in enumerate(records):
            self.attendance_table.setItem(row_idx, 0, QTableWidgetItem(str(id)))
            self.attendance_table.setItem(row_idx, 1, QTableWidgetItem(name))
            self.attendance_table.setItem(row_idx, 2, QTableWidgetItem(str(date)))
            self.attendance_table.setItem(row_idx, 3, QTableWidgetItem(str(first_login)))
            self.attendance_table.setItem(row_idx, 4, QTableWidgetItem(str(last_logout)))
            self.attendance_table.setItem(row_idx, 5, QTableWidgetItem(status))
            self.attendance_table.setItem(row_idx, 6, QTableWidgetItem(str(work_hours)))
            self.attendance_table.setItem(row_idx, 7, QTableWidgetItem(str(break_hours)))

            for col_idx in range(8):
                item = self.attendance_table.item(row_idx, col_idx)
                if item:
                    item.setTextAlignment(Qt.AlignCenter)
                    item.setBackground(QBrush(QColor(29, 53, 87)))  

    def update_attendance_data(self):
        # Call load_attendance_data without parameters to refresh the general view
        self.load_attendance_data()

    def on_cell_clicked(self, row, column):
        name = self.attendance_table.item(row, 1).text()  # Get name from column 1
        date = self.attendance_table.item(row, 2).text()  # Get date from column 2
        self.load_attendance_data(name=name, date=date)  # Pass both name and date to filter

    def hide_filters(self):
        """Hide the search and date filter widgets."""
        self.search_line_edit.setVisible(False)
        self.date_label.setVisible(False)
        self.date_edit.setVisible(False)
        self.date_filter_button.setVisible(False)

    def show_full_attendance(self):
        self.load_attendance_data()  # Reloads all attendance data without filters

        # Reset the toggle button to normal state
        self.date_filter_button.setText("Date Filter")
        self.date_edit.setVisible(False)
        self.date_label.setVisible(False)

        self.back_button.setVisible(False)  # Hide the back buttont

    def show_filters(self):
        """Show the search and date filter widgets."""
        self.search_line_edit.setVisible(True)
        self.date_label.setVisible(True)
        self.date_edit.setVisible(True)
        self.date_filter_button.setVisible(True)

    def download_report(self):
        # Open save dialog for the file path
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Attendance Report", "", "Excel Files (*.xlsx)", options=options)
        if file_path:
            # Create a Pandas Excel writer using Openpyxl as the engine
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Fetch attendance data
                attendance_df = pd.read_sql('''SELECT id, name, first_login_time AS "Login Time", last_logout_time AS "Log-Out Time", current_status AS "Status", total_work_hours AS "Working Hours", total_break_hours AS "Total Break Time" FROM attendance''', conn)
                # Write attendance DataFrame to a sheet
                attendance_df.to_excel(writer, sheet_name='Attendance', index=False)

                # Fetch attendance sessions data
                sessions_df = pd.read_sql('''SELECT attendance_sessions.attendance_id AS "Attendance ID", login_time AS "Login Time", logout_time AS "Logout Time" 
                                            FROM attendance_sessions 
                                            JOIN attendance ON attendance.id = attendance_sessions.attendance_id''', conn)
                # Write sessions DataFrame to a separate sheet
                sessions_df.to_excel(writer, sheet_name='Attendance Sessions', index=False)

            QMessageBox.information(self, "Report Saved", "Attendance report saved successfully.")

    def mousePressEvent(self, event):
        self.old_pos = event.globalPos()

    def mouseMoveEvent(self, event):
        delta = QPoint(event.globalPos() - self.old_pos)
        self.move(self.x() + delta.x(), self.y() + delta.y())
        self.old_pos = event.globalPos()

    def closeEvent(self, event):
        global stop_threads
        stop_threads = True
        time.sleep(0.5)

        if self.capture_thread.is_alive():
            self.capture_thread.join()
        if self.process_thread.is_alive():
            self.process_thread.join()

        self.cap.release()
        close_db_connection()

        event.accept()

    def update_attendance_table(self):
        self.attendance_table.setRowCount(0)
        conn = get_db_connection()
        cursor = conn.cursor()

        # Query the data to display on the table
        cursor.execute("""
            SELECT id, name, first_login_time, last_logout_time, current_status, total_work_hours, total_break_hours from attendance
        """)
        records = cursor.fetchall()

        for row_number, row_data in enumerate(records):
            self.attendance_table.insertRow(row_number)
            for column_number, data in enumerate(row_data):
                self.attendance_table.setItem(row_number, column_number, QTableWidgetItem(str(data)))

        conn.commit()

    def update_video_feed(self):
        if not result_queue.empty():
            frame, bboxes, landmarks, results, score_100 = result_queue.get()
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image)
            font_path = resource_path('utils/Roboto-Regular.ttf')
            font = ImageFont.truetype(font_path, 30)

            if bboxes is not None:
                for i, b in enumerate(bboxes):
                    if results[i] == -1:  # If the face is not recognized
                        triangle = [(b[0], b[3]), (b[2], b[3]), ((b[0] + b[2]) / 2, b[1])]
                        draw.polygon(triangle, outline='red', fill=None, width=5)
                        text_position = ((b[0] + b[2]) / 2, b[3] + 5)
                        draw.text(text_position, 'Unknown', fill=(255, 0, 0), font=font)
                    else:  # If the face is recognized
                        draw.rectangle([b[0], b[1], b[2], b[3]], outline='green', width=5)
                        name_position = (b[0], b[1] - 30)
                        draw.text(name_position, self.names[results[i] + 1], fill=(0, 255, 0), font=font)

            image = np.array(image)
            h, w, ch = image.shape
            bytes_per_line = ch * w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            self.face_recognition_label.setPixmap(pixmap.scaled(self.face_recognition_label.size(), Qt.KeepAspectRatio))

def main(args):
    cap = cv2.VideoCapture(camera_index)  # Open the camera
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load the model and facebank
    weight_path = resource_path('Weights/MobileFace_Net')
    detect_model = MobileFaceNet(512).to(device)
    detect_model.load_state_dict(torch.load(weight_path, map_location=lambda storage, loc: storage, weights_only=True))
    detect_model.eval()

    face_bank_path = resource_path_facebank('facebank')
    if args.update:
        targets, names = prepare_facebank(detect_model, path=face_bank_path, tta=args.tta)
    else:
        targets, names = load_facebank(path=face_bank_path)

    # Initialize the application and threads
    app = QApplication(sys.argv)
    capture_thread = threading.Thread(target=capture_frames, args=(cap,))
    process_thread = threading.Thread(target=process_frames, args=(args, device, detect_model, targets, names))

    # Start threads
    capture_thread.start()
    process_thread.start()

    window = AttendanceUI(names, capture_thread, process_thread, cap)
    window.setFixedSize(800, 900)
    window.show()

    # Handle application exit
    sys.exit(app.exec_())

    # Clean up resources after app exit
    capture_thread.join()
    process_thread.join()

    cap.release()  # camera is released
    QApplication.quit()  # quit the application
    event.accept()
    close_db_connection()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face detection demo')
    parser.add_argument('-th', '--threshold', help='Threshold score to decide identical faces', default=60, type=float)
    parser.add_argument("-u", "--update", help="Whether perform update the facebank", action="store_true", default=False)
    parser.add_argument("-tta", "--tta", help="Whether test time augmentation", action="store_true", default=False)
    parser.add_argument("-c", "--score", help="Whether show the confidence score", action="store_true", default=True)
    parser.add_argument("--scale", dest='scale', help="Input frame scale to adjust the speed", default=0.5, type=float)
    parser.add_argument('--mini_face', dest='mini_face', help="Minimum face size to be detected", default=40, type=int)
    parser.add_argument('--process_every', dest='process_every', help="Process every nth frame", default=3, type=int)
    parser.add_argument("-ci", "--camera_index", help="Camera index (default is 0)", default=0, type=str)
    args = parser.parse_args()

    def get_settings_file_path():
        # Check if running as a standalone executable
        if getattr(sys, 'frozen', False):
            # If the script is running as an executable, use the temporary folder
            current_directory = os.path.dirname(sys.executable)
        else:
            # If running as a script, use the script's directory
            current_directory = os.path.dirname(os.path.abspath(__file__))
        
        return os.path.join(current_directory, "camera_settings.txt")

    def save_camera_setting(setting):
        with open(get_settings_file_path(), "w") as f:
            f.write(setting)

    def load_camera_setting():
        settings_file = get_settings_file_path()
        if os.path.exists(settings_file):
            with open(settings_file, "r") as f:
                return f.read().strip()
        return None

    def get_camera_source():
        saved_source = load_camera_setting()
        if saved_source:
            print(f"Press Enter to use the last time used camera index/RTSP link: {saved_source}")
            source = input(f"Enter camera index or RTSP link (default index is 0): ").strip()
            source = source if source else saved_source  # Use saved setting if no input is provided
        else:
            source = input("Enter camera index or RTSP link (default index is 0): ").strip() or "0"

        # Save the new choice for future use
        save_camera_setting(source)  

        # Convert the source to an integer if it's a camera index
        try:
            return int(source) 
        except ValueError:
            return source  

    # Get the camera source
    camera_source = get_camera_source()

    # Set the camera index or RTSP link
    camera_index = camera_source

    main(args)