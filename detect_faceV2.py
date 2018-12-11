# 添加了人脸识别结果和匹配度

import sys
import cv2
import os
import datetime
import dlib
import math
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from faceClassifier import FaceClassifier
import numpy as np

facePath = './person_faces/'


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.timer_camera = QTimer()  # 需要定时器刷新摄像头界面
        self.cap = cv2.VideoCapture()
        self.cap_num = 0
        self.set_ui()  # 初始化UI界面
        self.slot_init()  # 初始化信号槽
        self.detect_flag = 0  # 人脸检测开关变量
        self.cap_face = None
        self.detector = FaceClassifier('dlib')

    def set_ui(self):
        # 布局设置
        self.layout_main = QHBoxLayout()  # 整体框架是水平布局
        self.layout_button = QVBoxLayout()
        self.layout_persons = QVBoxLayout()
        self.layout_show = QVBoxLayout()
        self.layout_identitys = QVBoxLayout()
        self.layout_texts = QVBoxLayout()

        self.r_w = 1
        self.r_h = 1

        # 捕获到的人脸显示设置
        self.w_p = int(100*self.r_w)
        self.h_p = int(100*self.r_h)
        self.labels_persons = []
        self.label_person1 = QLabel()
        self.label_person1.setFixedSize(self.w_p,self.h_p)
        self.labels_persons.append(self.label_person1)
        self.label_person2 = QLabel()
        self.label_person2.setFixedSize(self.w_p,self.h_p)
        self.labels_persons.append(self.label_person2)
        self.label_person3 = QLabel()
        self.label_person3.setFixedSize(self.w_p,self.h_p)
        self.labels_persons.append(self.label_person3)

        # 身份识别结果图片显示设置
        self.w_i = int(80*self.r_w)
        self.h_i = int(100*self.r_h)
        self.labels_identitys = []
        self.label_identity1 = QLabel()
        self.label_identity1.setFixedSize(self.w_i,self.h_i)
        self.labels_identitys.append(self.label_identity1)
        self.label_identity2 = QLabel()
        self.label_identity2.setFixedSize(self.w_i,self.h_i)
        self.labels_identitys.append(self.label_identity2)
        self.label_identity3 = QLabel()
        self.label_identity3.setFixedSize(self.w_i,self.h_i)
        self.labels_identitys.append(self.label_identity3)

        # 身份识别结果文字显示设置
        self.w_t = int(120*self.r_h)
        self.h_t = int(100*self.r_h)
        self.labels_texts = []
        self.label_text1 = QLabel()
        self.label_text1.setFixedSize(self.w_t,self.h_t)
        self.label_text1.setStyleSheet("QLabel{font-size:18px;font-style:Black}")
        self.labels_texts.append(self.label_text1)

        self.label_text2 = QLabel()
        self.label_text2.setFixedSize(self.w_t,self.h_t)
        self.label_text2.setStyleSheet("QLabel{font-size:18px;font-style:Black}")
        self.labels_texts.append(self.label_text2)

        self.label_text3 = QLabel()
        self.label_text3.setFixedSize(self.w_t,self.h_t)
        self.label_text3.setStyleSheet("QLabel{font-size:18px;font-style:Black}")
        self.labels_texts.append(self.label_text3)
        
        # 按钮设置
        
        style_button = "QPushButton{font-size:20px;color:blue}"
        self.btn_open_cam = QPushButton('打开相机')        # self.btn_open_cam.move(10, 10)
        self.btn_open_cam.setFixedSize(100*self.r_w, 45*self.r_h)
        self.btn_open_cam.setStyleSheet(style_button)

        self.btn_detection_face = QPushButton('人脸检测')
        self.btn_detection_face.setFixedSize(100*self.r_w, 45*self.r_h)
        self.btn_detection_face.setStyleSheet(style_button)

        self.btn_capture_face = QPushButton('人脸捕获')
        self.btn_capture_face.setFixedSize(100*self.r_w, 45*self.r_h)
        self.btn_capture_face.setStyleSheet(style_button)

        self.btn_save_face = QPushButton('人脸保存')
        self.btn_save_face.setFixedSize(100*self.r_w, 45*self.r_h)
        self.btn_save_face.setStyleSheet(style_button)

        self.quit = QPushButton('退出')
        self.quit.setFixedSize(100*self.r_w, 45*self.r_h)
        self.quit.setStyleSheet(style_button)

        self.name_label = QLabel('输入名字')
        self.name_label.setFixedSize(100*self.r_w, 40*self.r_h)
        self.name_label.setStyleSheet("QLabel{font-size:20px;}")

        self.name_input = QTextEdit() #名字输入框
        self.name_input.setFixedSize(100*self.r_w,50*self.r_h)
        self.name_input.setStyleSheet("QTextEdit{font-size:20px;}")
        # self.btn_close_cam.move(10, 30)

        # 显示视频
        self.title_label = QLabel('                 智能人脸识别系统')
        self.title_label.setFixedSize(640*self.r_w, 40*self.r_h)
        self.title_label.setStyleSheet("text-align:right; font-size:30px; font-weight:bold; font-style:Courier; color:red;")

        self.label_show_camera = QLabel()
        self.label_move = QLabel()
        self.label_save_face = QLabel()
        self.label_move.setFixedSize(100*self.r_w, 50*self.r_h)
        
        self.label_show_camera.setFixedSize(641*self.r_w, 481*self.r_h)
        self.label_show_camera.setAutoFillBackground(False)

        self.label_save_face.setFixedSize(80*self.r_w, 100*self.r_h)
        self.label_save_face.setAutoFillBackground(False)

        # 布局
        self.layout_button.addWidget(self.btn_open_cam)
        self.layout_button.addWidget(self.btn_detection_face)
        self.layout_button.addWidget(self.btn_capture_face)
        self.layout_button.addWidget(self.btn_save_face)
        self.layout_button.addWidget(self.quit)
        self.layout_button.addWidget(self.name_label)
        self.layout_button.addWidget(self.name_input)
        self.layout_button.addWidget(self.label_save_face)
        # self.layout_button.addWidget(self.label_move)

        self.layout_persons.addWidget(self.label_person1)
        self.layout_persons.addWidget(self.label_person2)
        self.layout_persons.addWidget(self.label_person3)

        self.layout_identitys.addWidget(self.label_identity1)
        self.layout_identitys.addWidget(self.label_identity2)
        self.layout_identitys.addWidget(self.label_identity3)    
        
        self.layout_texts.addWidget(self.label_text1)   
        self.layout_texts.addWidget(self.label_text2)  
        self.layout_texts.addWidget(self.label_text3)   

        self.layout_show.addWidget(self.title_label)
        self.layout_show.addWidget(self.label_show_camera)        

        self.layout_main.addLayout(self.layout_button)
        self.layout_main.addLayout(self.layout_show)
        self.layout_main.addLayout(self.layout_persons)
        self.layout_main.addLayout(self.layout_identitys)
        self.layout_main.addLayout(self.layout_texts)
        
        

        self.setLayout(self.layout_main)
        self.setGeometry(500, 200, 640*self.r_w, 480*self.r_h)
        self.setWindowTitle("人脸识别软件")


    # 信号槽设置
    def slot_init(self):
        self.btn_open_cam.clicked.connect(self.btn_open_cam_click)
        self.btn_detection_face.clicked.connect(self.detect_face)
        self.btn_capture_face.clicked.connect(self.capture_face)
        self.btn_save_face.clicked.connect(self.save_face)
        self.timer_camera.timeout.connect(self.show_camera)
        self.quit.clicked.connect(self.close)

    def btn_open_cam_click(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.cap_num)
            # flag = self.cap.open("rtsp://admin:adminZQRCOS@192.168.1.64:554/h264/ch1/main/av_stream")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            if flag == False:
                msg = QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确", buttons=QMessageBox.Ok,
                                          defaultButton=QMessageBox.Ok)
            # if msg==QtGui.QMessageBox.Cancel:
            #                     pass
            else:
                self.timer_camera.start(30)

                self.btn_open_cam.setText(u'关闭相机')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.btn_open_cam.setText(u'打开相机')

    def show_camera(self):
        if self.detect_flag == 0:
            ret, self.image = self.cap.read()
            show = cv2.resize(self.image, (640, 480))
            show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 这里指的是显示原图
            # opencv 读取图片的样式，不能通过Qlabel进行显示，需要转换为Qimage QImage(uchar * data, int width,
            # int height, Format format, QImageCleanupFunction cleanupFunction = 0, void *cleanupInfo = 0)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            self.label_show_camera.setPixmap(QPixmap.fromImage(showImage))
        else:
            ret_1, self.image_1 = self.cap.read()
            if ret_1:
                detect_image, informations = self.detector.face_detect(self.image_1)
                self.label_show_camera.setPixmap(QPixmap.fromImage(detect_image))
                i = 0
                for infm in informations:
                    if i >= 3:
                        break
                    if infm[1]=='unknow':
                        continue

                    print((infm[3][0], infm[3][1], infm[3][2], infm[3][3]))
                    face_img = self.image_1[infm[3][1]:infm[3][3], infm[3][0]:infm[3][2]]
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    face_img = cv2.resize(face_img, (self.w_p, self.h_p))
                    Q_img = QImage(face_img.data, face_img.shape[1], face_img.shape[0], QImage.Format_RGB888)
                    self.labels_persons[i].setPixmap(QPixmap.fromImage(Q_img))
                    # cv2.imshow('face', face_img)
                    # cv2.waitKey(10)

                    imgpath = facePath + infm[1] + '/' + infm[0]
                    img = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.w_i, self.h_i))
                    Q_img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
                    self.labels_identitys[i].setPixmap(QPixmap.fromImage(Q_img))
                    
                    ex = math.exp(1.0/infm[2] - 0.3)
                    math_tate = ex / (1+ex)
                    text_str = '姓名： ' + infm[1] + '\n' + '匹配度： ' + str(math_tate)[2:4] + '%'
                    self.labels_texts[i].setText(text_str)

                    i += 1
                while i<3:
                    self.labels_persons[i].setPixmap(QPixmap(""))
                    self.labels_identitys[i].setPixmap(QPixmap(""))
                    self.labels_texts[i].setText(' ')
                    i +=1
            else:
                print("please open camera!!!")


    def detect_face(self):
        if self.detect_flag == 0:
            self.detect_flag = 1
            self.btn_detection_face.setText(u'关闭检测')
        else:
            self.detect_flag = 0
            self.btn_detection_face.setText(u'人脸检测')


    def capture_face(self):
        self.cap_face = self.detector.detected_face  # BGR cv格式  捕获到的人脸
        self.cap_portrait = self.detector.detected_portrait  # 捕获到的头像
        show_face = cv2.cvtColor(self.cap_portrait, cv2.COLOR_BGR2RGB)
        show_face = cv2.resize(show_face, (self.w_i,self.h_i))
        show_face = QImage(show_face.data, show_face.shape[1], show_face.shape[0], QImage.Format_RGB888)
        self.label_save_face.setPixmap(QPixmap.fromImage(show_face))


    def save_face(self):
        label = self.name_input.toPlainText()
        print(label)
        # 判断输入框中是否有异常字符
        if label:
            if not os.path.exists(facePath + label):
                os.makedirs(facePath + label)
            time_now = datetime.datetime.now().strftime('%H%M')
            imgSavePath = facePath + label + '/' + label + '_' + time_now + '.jpg'
            print(imgSavePath)
            cv2.imencode('.jpg', self.cap_portrait)[1].tofile(imgSavePath)
            # cv2.imwrite(imgSavePath, self.cap_portrait)    # 保存的图片是头像

            save_flag = self.detector.face_save(self.cap_face, label) # 保存的特征是人脸的特征
            if save_flag:
                self.name_input.setPlainText('')
                self.label_save_face.clear()
        else:
            print('请输入正确字符！！！')


    def closeEvent(self, QCloseEvent):

        reply = QMessageBox.question(self, u"Warning", "Are you sure quit ?", QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.cap.release()
            self.timer_camera.stop()
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
    print('end')