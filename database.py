import mysql.connector
from mysql.connector import Error
from datetime import datetime

class Database:
    def __init__(self, host="localhost", port=1025, user="root", password="root", database="emoji_tracker"):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.conn = None
        self.cursor = None

    def connect(self):
        try:
            # Connect without specifying database to create it if missing
            temp_conn = mysql.connector.connect(host=self.host, user=self.user, password=self.password)
            temp_cursor = temp_conn.cursor()
            temp_cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            temp_conn.close()

            # Connect to DB
            self.conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            self.cursor = self.conn.cursor()
            print("Connected to MySQL database.")

            # Create (legacy) moods table if you still use it (keeps backwards compatibility)
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS moods (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    mood VARCHAR(255),
                    confidence FLOAT,
                    duration FLOAT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.commit()

            # Create the new face_moods table (for person + emotion)
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_moods (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    person_name VARCHAR(100),
                    emotion VARCHAR(50),
                    confidence FLOAT,
                    duration FLOAT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.commit()

        except Error as err:
            print(f"Database connection error: {err}")
            self.conn = None

    def log_mood(self, mood, confidence=0.0, duration=0.0, timestamp=None):
        """Legacy: logs mood into moods table (keeps compatibility)."""
        if not self.conn:
            print("No active database connection. Skipping log.")
            return
        if timestamp is None:
            timestamp = datetime.now()
        try:
            confidence = float(confidence)
            duration = float(duration)
            sql = "INSERT INTO moods (mood, confidence, duration, timestamp) VALUES (%s, %s, %s, %s)"
            self.cursor.execute(sql, (mood, confidence, duration, timestamp))
            self.conn.commit()
            # Don't spam prints in production; helpful for debugging
            print(f"Logged (moods): {mood} | conf={confidence:.2f} | dur={duration:.2f}")
        except Error as e:
            print(f"Failed to log mood: {e}")

    def log_face_mood(self, person_name, emotion, confidence=0.0, duration=0.0, timestamp=None):
        """Logs person + emotion into face_moods table."""
        if not self.conn:
            print("No active database connection. Skipping face_mood log.")
            return
        if timestamp is None:
            timestamp = datetime.now()
        try:
            confidence = float(confidence)
            duration = float(duration)
            sql = "INSERT INTO face_moods (person_name, emotion, confidence, duration, timestamp) VALUES (%s, %s, %s, %s, %s)"
            self.cursor.execute(sql, (person_name, emotion, confidence, duration, timestamp))
            self.conn.commit()
            print(f"Logged (face_moods): {person_name} | {emotion} | conf={confidence:.2f} | dur={duration:.2f}")
        except Error as e:
            print(f"Failed to log face_mood: {e}")

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print("MySQL connection closed. Bye :)")
