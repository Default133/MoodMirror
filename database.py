import mysql.connector
from mysql.connector import Error

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
            print(" Connected to MySQL database.")

            # Step 4: Create moods table if it doesn’t exist
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS moods (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    mood VARCHAR(50),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.commit()

        except Error as err:
            print(f" Database connection error: {err}")
            self.conn = None

    def log_mood(self, mood):
        """Logs a detected mood into the MySQL database"""
        if not self.conn:
            print(" No active database connection. Skipping log.")
            return
        try:
            self.cursor.execute("INSERT INTO moods (mood) VALUES (%s)", (mood,))
            self.conn.commit()
            print(f" Logged mood: {mood}")
        except Error as e:
            print(f" Failed to log mood: {e}")

    def close(self):
        """Safely close MySQL connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print("MySQL connection closed. Bye :)")
