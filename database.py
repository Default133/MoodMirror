# -*- coding: ascii -*-

import mysql.connector
from mysql.connector import Error
from datetime import datetime
import threading
import time
import logging

logging.basicConfig(level=logging.INFO)

class Database:
    def __init__(self, host="localhost", port=1025, user="root", password="root", database="emoji_tracker", reconnect_attempts=3, reconnect_delay=1.0):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.conn = None
        self.lock = threading.Lock()
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay

    def connect(self):
        attempt = 0
        while attempt < self.reconnect_attempts:
            try:
                temp_conn = mysql.connector.connect(host=self.host, user=self.user, password=self.password)
                temp_cursor = temp_conn.cursor()
                temp_cursor.execute("CREATE DATABASE IF NOT EXISTS %s" % (self.database,))
                temp_cursor.close()
                temp_conn.close()

                self.conn = mysql.connector.connect(
                    host=self.host,
                    user=self.user,
                    password=self.password,
                    database=self.database,
                    autocommit=False
                )

                with self.conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS moods (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            mood VARCHAR(255),
                            confidence FLOAT,
                            duration FLOAT,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    cur.execute("""
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
                logging.info("Connected to MySQL database.")
                return
            except Error as err:
                logging.warning("Database connect attempt %d failed: %s", attempt+1, err)
                self.conn = None
                attempt += 1
                time.sleep(self.reconnect_delay)
        logging.error("Exceeded maximum database connect attempts; continuing without DB connection.")

    def _ensure_connection(self):
        try:
            if self.conn is None or not self.conn.is_connected():
                self.connect()
        except Exception:
            self.connect()

    def _execute(self, sql, params):
        self._ensure_connection()
        if not self.conn:
            logging.debug("No DB connection available for execute.")
            return False
        try:
            with self.lock:
                cursor = self.conn.cursor()
                cursor.execute(sql, params)
                self.conn.commit()
                cursor.close()
            return True
        except Error as e:
            logging.error("DB execution failed: %s", e)
            try:
                if self.conn:
                    self.conn.rollback()
            except Exception:
                pass
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
            return False

    def _fetchall(self, sql, params=()):
        """
        Thread-safe fetch helper that returns a list of dict rows.
        """
        self._ensure_connection()
        if not self.conn:
            logging.debug("No DB connection available for fetch.")
            return []
        try:
            with self.lock:
                cursor = self.conn.cursor()
                cursor.execute(sql, params)
                columns = cursor.column_names
                rows = cursor.fetchall()
                cursor.close()
            results = []
            for row in rows:
                d = {}
                for i, col in enumerate(columns):
                    d[col] = row[i]
                results.append(d)
            return results
        except Error as e:
            logging.error("DB fetch failed: %s", e)
            try:
                if self.conn:
                    self.conn.rollback()
            except Exception:
                pass
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
            return []

    def log_mood(self, mood, confidence=0.0, duration=0.0, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()
        try:
            confidence = float(confidence)
            duration = float(duration)
        except Exception:
            confidence = 0.0
            duration = 0.0
        sql = "INSERT INTO moods (mood, confidence, duration, timestamp) VALUES (%s, %s, %s, %s)"
        success = self._execute(sql, (mood, confidence, duration, timestamp))
        if success:
            logging.debug("Logged (moods): %s | conf=%.2f | dur=%.2f", mood, confidence, duration)
        return success

    def log_face_mood(self, person_name, emotion, confidence=0.0, duration=0.0, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()
        try:
            confidence = float(confidence)
            duration = float(duration)
        except Exception:
            confidence = 0.0
            duration = 0.0
        sql = "INSERT INTO face_moods (person_name, emotion, confidence, duration, timestamp) VALUES (%s, %s, %s, %s, %s)"
        success = self._execute(sql, (person_name, emotion, confidence, duration, timestamp))
        if success:
            logging.debug("Logged (face_moods): %s | %s | conf=%.2f | dur=%.2f", person_name, emotion, confidence, duration)
        return success

    def get_recent_face_moods(self, minutes=60, limit=1000):
        """
        Returns recent face_moods within the last `minutes`.
        Result is a list of dicts with keys: person_name, emotion, confidence, duration, timestamp
        """
        try:
            sql = ("SELECT person_name, emotion, confidence, duration, timestamp "
                   "FROM face_moods WHERE timestamp >= DATE_SUB(NOW(), INTERVAL %s MINUTE) "
                   "ORDER BY timestamp DESC LIMIT %s")
            rows = self._fetchall(sql, (int(minutes), int(limit)))
            return rows
        except Exception as e:
            logging.error("Failed to get recent face_moods: %s", e)
            return []

    def close(self):
        try:
            if self.conn:
                self.conn.close()
                logging.info("MySQL connection closed.")
        except Exception:
            pass