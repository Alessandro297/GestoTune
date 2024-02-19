import sys

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from GUI.MainWindow import MainWindow

def main():
    # Handle resolution displays:
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, False)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, False)

    application = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(application.exec_())

if __name__ == "__main__":
    main()
