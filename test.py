# 添加刷卡功能

import sys
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from faceClassifier import FaceClassifier

facePath = './person_faces/'

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.set_ui()

    def set_ui(self):
        # 布局设置
        self.name_label = QLabel('输入名字', self)
        self.name_label.setFixedSize(100, 40)
        self.name_label.setStyleSheet("QLabel{font-size:20px;}")
        self.use_palette()
        
        self.setGeometry(500, 200, 640, 480)
    def use_palette(self): 
        self.setWindowTitle("设置背景图片") 
        window_pale = QPalette() 
        window_pale.setBrush(self.backgroundRole(), QBrush(QPixmap("5.jpg"))) 
        self.setPalette(window_pale)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
    print('end')