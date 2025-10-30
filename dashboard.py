# -*- coding: ascii -*-
"""
dashboard.py

A PyQt5 dashboard window that visualizes emotion data stored in the
emoji_tracker MySQL database (face_moods table).

Features:
- Daily emotion distribution pie chart (select date)
- Emotion over time line graph (last N hours or selected date)
- Top 3 most common moods summary
- Compare emotions by person (stacked bar chart for top persons)

Dependencies:
- PyQt5
- matplotlib
- numpy
- mysql-connector-python (used via your existing Database helper)

Usage:
- This file provides DashboardWindow, which can be imported and opened
  from your main GUI (MainWindow) with:
      from dashboard import DashboardWindow
      win = DashboardWindow(db_instance)
      win.show()
"""
from __future__ import annotations


import sys
import datetime
import logging

from PyQt5 import QtCore, QtGui, QtWidgets

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np


from database import Database

logging.basicConfig(level=logging.INFO)

def db_read(db: Database, sql: str, params: tuple=()):
    """
    Executes a read-only SQL query and returns rows as a list of tuples.
    Preference order:
      1) If Database has a _fetchall(sql, params) method, call it (returns list of dicts or tuples).
      2) Otherwise fall back to acquiring db.lock and using db.conn.cursor().
    Returns list of tuples (matching previous behavior).
    """
    # Try Database._fetchall if available
    try:
        if hasattr(db, "_fetchall") and callable(getattr(db, "_fetchall")):
            rows = db._fetchall(sql, params)
            # _fetchall in some implementations returns list of dicts; convert to tuples
            if rows and isinstance(rows[0], dict):
                # preserve column order by extracting values in cursor order isn't possible here,
                # so return tuples of values in dict value order (best-effort).
                return [tuple(r.values()) for r in rows]
            return rows
    except Exception:
        logging.debug("db._fetchall failed, falling back to direct cursor", exc_info=True)

    # Fallback: use db.conn with lock
    db._ensure_connection()
    if not getattr(db, "conn", None):
        return []
    try:
        with db.lock:
            cursor = db.conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            cursor.close()
        return rows
    except Exception as e:
        logging.error("DB read failed: %s", e)
        return []


class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        fig.tight_layout()

class DashboardWindow(QtWidgets.QWidget):
    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MoodMirror Dashboard")
        self.resize(1000, 700)
        self.db = db

        # Controls: date picker, person selector, refresh
        controls_layout = QtWidgets.QHBoxLayout()

        self.date_edit = QtWidgets.QDateEdit(QtCore.QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")

        self.person_combo = QtWidgets.QComboBox()
        self.person_combo.addItem("All persons")

        self.hours_spin = QtWidgets.QSpinBox()
        self.hours_spin.setRange(1, 168)
        self.hours_spin.setValue(24)
        self.hours_spin.setSuffix("h")

        refresh_btn = QtWidgets.QPushButton("Refresh")

        controls_layout.addWidget(QtWidgets.QLabel("Date:"))
        controls_layout.addWidget(self.date_edit)
        controls_layout.addSpacing(12)
        controls_layout.addWidget(QtWidgets.QLabel("Last:"))
        controls_layout.addWidget(self.hours_spin)
        controls_layout.addWidget(QtWidgets.QLabel("hours"))
        controls_layout.addSpacing(12)
        controls_layout.addWidget(QtWidgets.QLabel("Person:"))
        controls_layout.addWidget(self.person_combo)
        controls_layout.addStretch(1)
        controls_layout.addWidget(refresh_btn)

        # Charts: pie, time series, top3 text, compare by person
        self.pie_canvas = MplCanvas(width=4, height=3, dpi=100)
        self.time_canvas = MplCanvas(width=6, height=3, dpi=100)
        self.compare_canvas = MplCanvas(width=6, height=4, dpi=100)

        # Top 3 summary as plain text widget
        self.top3_widget = QtWidgets.QTextEdit()
        self.top3_widget.setReadOnly(True)
        self.top3_widget.setFixedHeight(120)

        # Layout arrangement
        top_layout = QtWidgets.QHBoxLayout()
        top_left = QtWidgets.QVBoxLayout()
        top_left.addWidget(QtWidgets.QLabel("Daily Emotion Distribution"))
        top_left.addWidget(self.pie_canvas)
        top_left.addWidget(QtWidgets.QLabel("Top 3 Most Common Moods"))
        top_left.addWidget(self.top3_widget)

        top_right = QtWidgets.QVBoxLayout()
        top_right.addWidget(QtWidgets.QLabel("Emotion Over Time"))
        top_right.addWidget(self.time_canvas)

        top_layout.addLayout(top_left, stretch=1)
        top_layout.addLayout(top_right, stretch=2)

        bottom_layout = QtWidgets.QVBoxLayout()
        bottom_layout.addWidget(QtWidgets.QLabel("Compare Emotions by Person"))
        bottom_layout.addWidget(self.compare_canvas)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(controls_layout)
        main_layout.addLayout(top_layout)
        main_layout.addLayout(bottom_layout)
        self.setLayout(main_layout)

        # Signals
        refresh_btn.clicked.connect(self.refresh)
        self.date_edit.dateChanged.connect(lambda _: self.refresh())
        self.person_combo.currentIndexChanged.connect(lambda _: self.refresh())

        # initial population
        self._load_persons()
        self.refresh()

    def _load_persons(self):
        """
        Populate person combo with distinct person_names from face_moods.
        """
        rows = db_read(self.db, "SELECT DISTINCT person_name FROM face_moods WHERE person_name IS NOT NULL AND person_name != ''")
        names = sorted([r[0] for r in rows if r[0]])
        self.person_combo.blockSignals(True)
        self.person_combo.clear()
        self.person_combo.addItem("All persons")
        for n in names:
            self.person_combo.addItem(n)
        self.person_combo.blockSignals(False)

    def refresh(self):
        """
        Recompute and redraw all charts using DB data.
        """
        try:
            self._draw_pie_chart()
            self._draw_time_series()
            self._draw_top3()
            self._draw_compare_by_person()
        except Exception as e:
            logging.error("Failed to refresh dashboard: %s", e)

    def _draw_pie_chart(self):
        """
        Daily emotion distribution for selected date.
        """
        qdate = self.date_edit.date().toPyDate()
        start = datetime.datetime.combine(qdate, datetime.time.min)
        end = datetime.datetime.combine(qdate, datetime.time.max)
        sql = "SELECT emotion, COUNT(*) FROM face_moods WHERE timestamp >= %s AND timestamp <= %s GROUP BY emotion"
        rows = db_read(self.db, sql, (start, end))
        labels = []
        sizes = []
        for r in rows:
            labels.append(r[0] if r[0] else "Unknown")
            sizes.append(int(r[1]))
        ax = self.pie_canvas.axes
        ax.clear()
        if sizes:
            ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
            ax.set_title("Distribution for %s" % (qdate.isoformat(),))
        else:
            ax.text(0.5, 0.5, "No data for selected date", horizontalalignment="center", verticalalignment="center")
            ax.set_title("Distribution for %s" % (qdate.isoformat(),))
        self.pie_canvas.draw()

    def _draw_time_series(self):
        """
        Emotion over time: count per time bucket over last N hours (or selected date).
        If a person is selected, show for that person only; otherwise aggregate all.
        """
        hours = int(self.hours_spin.value())
        end = datetime.datetime.now()
        start = end - datetime.timedelta(hours=hours)

        person = None
        if self.person_combo.currentIndex() > 0:
            person = self.person_combo.currentText()

        # We'll build a pivot: timeseries per emotion (hourly)
        sql = """
            SELECT DATE_FORMAT(timestamp, '%%Y-%%m-%%d %%H:00:00') as bucket,
                   emotion, COUNT(*) as cnt
            FROM face_moods
            WHERE timestamp >= %s AND timestamp <= %s
        """
        params = [start, end]
        if person:
            sql += " AND person_name = %s"
            params.append(person)
        sql += " GROUP BY bucket, emotion ORDER BY bucket ASC"
        rows = db_read(self.db, sql, tuple(params))

        # Build time buckets
        buckets = []
        t = start.replace(minute=0, second=0, microsecond=0)
        while t <= end:
            buckets.append(t)
            t += datetime.timedelta(hours=1)
        bucket_strs = [dt.strftime("%Y-%m-%d %H:00:00") for dt in buckets]

        # gather emotions
        emotions = {}
        for b, emo, cnt in rows:
            if emo not in emotions:
                emotions[emo] = {bs: 0 for bs in bucket_strs}
            emotions[emo][b] = int(cnt)

        ax = self.time_canvas.axes
        ax.clear()
        if emotions:
            for emo, data in emotions.items():
                y = [data.get(bs, 0) for bs in bucket_strs]
                ax.plot(bucket_strs, y, label=str(emo))
            ax.set_xticks(bucket_strs[::max(1, len(bucket_strs)//8)])
            ax.set_xticklabels(bucket_strs[::max(1, len(bucket_strs)//8)], rotation=45, ha="right")
            ax.set_title("Emotion counts over last %d hours%s" % (hours, (" for " + person) if person else ""))
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No data for selected period", horizontalalignment="center", verticalalignment="center")
            ax.set_title("Emotion counts over last %d hours" % (hours,))
        self.time_canvas.draw()

    def _draw_top3(self):
        """
        Query top 3 moods overall (all time or filtered by date).
        """
        qdate = self.date_edit.date().toPyDate()
        start = datetime.datetime.combine(qdate, datetime.time.min)
        end = datetime.datetime.combine(qdate, datetime.time.max)
        sql = "SELECT emotion, COUNT(*) as cnt FROM face_moods WHERE timestamp >= %s AND timestamp <= %s GROUP BY emotion ORDER BY cnt DESC LIMIT 3"
        rows = db_read(self.db, sql, (start, end))
        lines = []
        total = 0
        for r in rows:
            lines.append("%s: %d" % (r[0] if r[0] else "Unknown", int(r[1])))
            total += int(r[1])
        if not lines:
            text = "No data for selected date"
        else:
            text = "Top 3 moods for %s\n\n" % (qdate.isoformat(),)
            text += "\n".join(lines)
            text += "\n\nTotal (top3): %d" % (total,)
        self.top3_widget.setPlainText(text)

    def _draw_compare_by_person(self):
        """
        Compare emotions by person: for top persons by number of entries, draw stacked bar.
        """
        # find top persons
        sql_persons = "SELECT person_name, COUNT(*) as cnt FROM face_moods WHERE person_name IS NOT NULL AND person_name != '' GROUP BY person_name ORDER BY cnt DESC LIMIT 5"
        rows = db_read(self.db, sql_persons)
        persons = [r[0] for r in rows if r[0]]
        if not persons:
            ax = self.compare_canvas.axes
            ax.clear()
            ax.text(0.5, 0.5, "No person-labeled data available", horizontalalignment="center", verticalalignment="center")
            self.compare_canvas.draw()
            return

        # Get emotion counts per person
        placeholders = ", ".join(["%s"] * len(persons))
        sql = f"SELECT person_name, emotion, COUNT(*) FROM face_moods WHERE person_name IN ({placeholders}) GROUP BY person_name, emotion"
        rows = db_read(self.db, sql, tuple(persons))

        # Build pivot table
        emotions = sorted(list({r[1] for r in rows if r[1]}))
        if not emotions:
            # fallback: use counts per person without emotion categories
            ax = self.compare_canvas.axes
            ax.clear()
            counts = [r[0] for r in rows]
            ax.bar(persons, [1]*len(persons))
            ax.set_title("No emotion categories available")
            self.compare_canvas.draw()
            return

        data = {emo: [0] * len(persons) for emo in emotions}
        for person_name, emo, cnt in rows:
            if person_name in persons:
                i = persons.index(person_name)
                data[emo][i] = int(cnt)

        # stacked bar chart
        ax = self.compare_canvas.axes
        ax.clear()
        ind = np.arange(len(persons))
        bottom = np.zeros(len(persons), dtype=int)
        cmap = matplotlib.cm.get_cmap("tab20")
        for i, emo in enumerate(emotions):
            vals = data[emo]
            ax.bar(ind, vals, bottom=bottom, label=str(emo), color=cmap(i % 20))
            bottom = bottom + np.array(vals)
        ax.set_xticks(ind)
        ax.set_xticklabels(persons, rotation=45, ha="right")
        ax.set_title("Emotion distribution for top persons")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        self.compare_canvas.draw()

# Standalone run for testing
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    db = Database(host="localhost", port=1025, user="root", password="root", database="emoji_tracker")
    db.connect()
    w = DashboardWindow(db)
    w.show()
    sys.exit(app.exec_())