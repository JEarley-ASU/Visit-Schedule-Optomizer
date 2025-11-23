import sys
from PySide6.QtWidgets import QApplication
from Main_window import MeetingScheduler

if __name__ == "__main__":
    app = QApplication(sys.argv)
    scheduler = MeetingScheduler()
    scheduler.show()  # Change from exec() to show()
    sys.exit(app.exec()) 