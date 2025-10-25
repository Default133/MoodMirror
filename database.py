import mysql.connector

class Database:
    def _init_(self, host="localhost",port=1025, user="root", password="root", database="emoji_tracker"):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.conn = None
        self.cursor = None

    def connect(self):
        try:
            self.conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            self.cursor = self.conn.cursor()
            print(" Connected to MySQL database.")
            self.cursor.execute("""CREATE TABLE IF NOT EXISTS moods ( id INT AUTO_INCREMENT PRIMARY KEY, mood VARCHAR(50), timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")

        except mysql.connector.Error as err:
            print(f" Database connection error: {err}")
            self.conn = None

    def log_mood(self, mood):
        if not self.conn:
            print(" No active database connection. Skipping log.")
            return
        try:
            self.cursor.execute("INSERT INTO moods (mood) VALUES (%s)", (mood,))
            self.conn.commit()
            print(f" Logged mood: {mood}")
        except Exception as e:
            print(f" Failed to log mood: {e}")

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print(" MySQL connection closed.Bye :)")
