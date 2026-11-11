import os
import cv2
import sqlite3
import smtplib
import threading
from pathlib import Path
from datetime import datetime
from email.message import EmailMessage
from dotenv import load_dotenv


load_dotenv()


class AlertManager:

    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()
        self.last_alert = {}

        Path("incidents/snapshots").mkdir(
            parents=True, exist_ok=True
        )

        self.db_path = "incidents/incidents.db"
        self._initialize_database()

    def _initialize_database(self):
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera TEXT,
                    person_id INTEGER,
                    timestamp TEXT,
                    snapshot TEXT
                )
            """)

    def report_fall(self, camera, person_id, frame):
        key = (camera, person_id)
        now = datetime.now()

        cooldown = self.config["alerts"]["cooldown_seconds"]

        with self.lock:
            previous = self.last_alert.get(key)

            if previous is not None:
                elapsed = (now - previous).total_seconds()

                if elapsed < cooldown:
                    return

            self.last_alert[key] = now

        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = (
            f"incidents/snapshots/"
            f"{camera.replace(' ', '_')}_"
            f"person_{person_id}_{timestamp}.jpg"
        )

        if self.config["alerts"]["save_snapshots"]:
            cv2.imwrite(filename, frame)
        else:
            filename = ""

        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO incidents
                (camera, person_id, timestamp, snapshot)
                VALUES (?, ?, ?, ?)
                """,
                (
                    camera,
                    person_id,
                    now.isoformat(timespec="seconds"),
                    filename
                )
            )

        print(
            f"[ALERT] Fall detected: {camera}, "
            f"Person {person_id}"
        )

        email_config = self.config["alerts"]["email"]

        if email_config["enabled"]:
            threading.Thread(
                target=self._send_email,
                args=(camera, person_id, filename, now),
                daemon=True
            ).start()

    def _send_email(self, camera, person_id, snapshot, timestamp):
        settings = self.config["alerts"]["email"]

        username = os.getenv("SMTP_USERNAME")
        password = os.getenv("SMTP_PASSWORD")

        if not username or not password:
            print("[EMAIL] SMTP credentials missing.")
            return

        message = EmailMessage()
        message["Subject"] = "CCTV Fall Detection Alert"
        message["From"] = settings["sender"] or username
        message["To"] = settings["recipient"]

        message.set_content(
            f"""
A possible fall has been detected.

Camera: {camera}
Person ID: {person_id}
Time: {timestamp.strftime("%Y-%m-%d %H:%M:%S")}

Please review the CCTV footage.
"""
        )

        if snapshot and os.path.exists(snapshot):
            with open(snapshot, "rb") as file:
                message.add_attachment(
                    file.read(),
                    maintype="image",
                    subtype="jpeg",
                    filename=os.path.basename(snapshot)
                )

        try:
            with smtplib.SMTP(
                settings["smtp_server"],
                settings["smtp_port"],
                timeout=15
            ) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(message)

            print("[EMAIL] Alert sent successfully.")

        except Exception as error:
            print(f"[EMAIL ERROR] {error}")