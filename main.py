import sys

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from GUI.MainWindow import MainWindow

def main():
    # Handle high resolution displays:
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    application = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(application.exec_())

if __name__ == "__main__":
    main()
