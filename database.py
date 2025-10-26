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
            # Step 1: Connect without specifying a database
            temp_conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password
            )
            temp_cursor = temp_conn.cursor()

            # Step 2: Create the database if it doesn't exist
            temp_cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            temp_conn.close()

            # Step 3: Connect to the actual database
            self.conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            self.cursor = self.conn.cursor()
            print("Connected to MySQL database.")

            # Step 4: Create moods table if it doesn’t exist
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS moods (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    mood VARCHAR(50),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.commit()

            # Step 5: Automatically add missing columns
            self._ensure_column("moods", "confidence", "FLOAT", default=0)
            self._ensure_column("moods", "duration", "FLOAT", default=0)

        except Error as err:
            print(f"Database connection error: {err}")
            self.conn = None

    def _ensure_column(self, table_name, column_name, column_type, default=None):
        """Add a column to the table if it does not exist."""
        self.cursor.execute(f"SHOW COLUMNS FROM {table_name} LIKE '{column_name}'")
        result = self.cursor.fetchone()
        if not result:
            default_sql = f"DEFAULT {default}" if default is not None else ""
            sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type} {default_sql}"
            self.cursor.execute(sql)
            self.conn.commit()
            print(f"Added missing column '{column_name}' to '{table_name}'")

    def log_mood(self, mood, confidence=0.0, duration=0.0, timestamp=None):
        """Logs a detected mood into the MySQL database"""
        if not self.conn:
            print("No active database connection. Skipping log.")
            return

        if timestamp is None:
            timestamp = datetime.now()

        try:
            # Ensure numpy types are converted to native floats
            confidence = float(confidence)
            duration = float(duration)

            sql = "INSERT INTO moods (mood, confidence, duration, timestamp) VALUES (%s, %s, %s, %s)"
            self.cursor.execute(sql, (mood, confidence, duration, timestamp))
            self.conn.commit()
            print(f"Logged mood: {mood}, confidence: {confidence}, duration: {duration}, timestamp: {timestamp}")
        except Error as e:
            print(f"Failed to log mood: {e}")

    def close(self):
        """Safely close MySQL connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print("MySQL connection closed. Bye :)")
