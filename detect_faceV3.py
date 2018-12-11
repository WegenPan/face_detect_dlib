# 添加刷卡功能

import sys
import cv2
import os
import datetime
import dlib
import math
import serial
import binascii
import time
import threading
import numpy as np
from ctypes import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from faceClassifier import FaceClassifier

facePath = './person_faces/'


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.timer_camera = QTimer()  # 需要定时器刷新摄像头界面
        self.timer_serial = QTimer()  # 定时器读取IC设备
        self.timer_ID = QTimer()      # 定时器读取ID设备
        self.cap = cv2.VideoCapture()
        self.cap_num = 0
        self.set_ui()  # 初始化UI界面
        self.slot_init()  # 初始化信号槽
        self.detect_flag = 0  # 人脸检测开关变量
        self.cap_face = None
        self.ID_ifo = None
        self.ID_ready = False
        self.ID_detector = None
        self.detector = FaceClassifier('dlib')
        self.ID_detect_init()

    def set_ui(self):
        # 布局设置
        self.layout_main = QHBoxLayout()  # 整体框架是水平布局
        self.layout_button = QVBoxLayout()
        self.layout_show = QVBoxLayout()
        self.layout_face_IC = QVBoxLayout()
        self.layout_faces = QHBoxLayout()
        self.layout_IC = QVBoxLayout()
        self.layout_persons = QVBoxLayout()
        self.layout_identitys = QVBoxLayout()
        self.layout_texts = QVBoxLayout()

        self.r_w = 1
        self.r_h = 1

        # 身份识别结果图片显示设置
        self.w_i = int(80*self.r_w)
        self.h_i = int(100*self.r_h)
        self.labels_identitys = []
        self.label_identity1 = QLabel()
        self.label_identity1.setFixedSize(self.w_i,self.h_i)
        self.labels_identitys.append(self.label_identity1)

        # 身份识别结果文字显示设置、IC功能显示设置
        self.label_face_IC = QLabel(self)
        self.label_face_IC.setFixedSize(240*self.r_w, 480*self.r_h)
        self.label_face_IC.setAutoFillBackground(True)
        self.label_face_IC.setStyleSheet("QLabel{background-color:rgb(50, 50, 50, 80);}")
        self.label_face_IC.move(580, 0)

        self.w_t = int(120*self.r_h)
        self.h_t = int(100*self.r_h)
        self.labels_texts = []
        self.label_text1 = QLabel()
        self.label_text1.setFixedSize(self.w_t,self.h_t)
        self.label_text1.setStyleSheet("QLabel{font-size:18px;font-style:Black}")
        self.labels_texts.append(self.label_text1)


        style_button_IC = "QPushButton{font-size:15px;color:blue;background-color:rgb(100, 100, 100, 50);}"
        self.btn_IC = QPushButton('打开IC功能')        # self.btn_open_cam.move(10, 10)
        self.btn_IC.setFixedSize(100*self.r_w, 30*self.r_h)
        self.btn_IC.setStyleSheet(style_button_IC) 

        self.label_ic = QLabel('')
        self.label_ic.setFixedSize(150*self.r_w, 30*self.r_h)
        self.label_ic.setStyleSheet("QLabel{font-size:15px;background:white;background-color:rgb(100, 100, 100, 50);}")

        # 按钮设置
        
        self.label_btn = QLabel(self)
        self.label_btn.setFixedSize(120*self.r_w, 450*self.r_h)
        self.label_btn.setAutoFillBackground(True)
        self.label_btn.setStyleSheet("QLabel{background-color:rgb(250, 250, 250, 80);}")
        self.label_btn.move(0, 0)

        style_button = "QPushButton{font-size:20px;background: transparent;}"
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
        self.name_input.setStyleSheet("QTextEdit{font-size:20px;background-color:rgb(100, 100, 100, 50);}")
        # self.btn_close_cam.move(10, 30)

        # 显示视频
        self.title_label = QLabel('')
        self.title_label.setFixedSize(480*self.r_w, 200*self.r_h)
        self.title_label.setStyleSheet("text-align:center; font-size:30px; font-weight:bold; font-style:Courier; color:red;")

        self.label_show_camera = QLabel()
        self.label_move = QLabel()
        self.label_save_face = QLabel()
        self.label_move.setFixedSize(100*self.r_w, 50*self.r_h)
        
        self.label_show_camera.setFixedSize(320*self.r_w, 240*self.r_h)
        self.label_show_camera.setAutoFillBackground(True)
        self.label_show_camera.setStyleSheet("QLabel{background-color:rgb(100, 100, 100, 50);border:2px solid rgb(0, 100, 150);}")

        self.label_save_face.setFixedSize(80*self.r_w, 100*self.r_h)
        self.label_save_face.setAutoFillBackground(False)

        # 身份证核验
        self.label_IDText = QLabel("身份证核验", self)
        self.label_IDText.setFixedSize(240, 20)
        self.label_IDText.setStyleSheet("QLabel{text-align:center;font-size:15px;background-color:rgb(150, 150, 150, 50)}")
        self.label_IDText.setAlignment(Qt.AlignCenter)

        self.label_IDText.move(580, 150)
        self.label_IDImg = QLabel(self)
        self.label_IDImg.setFixedSize(102, 126)
        self.label_IDImg.setStyleSheet("QLabel{background-color:rgb(100, 100, 100, 50);border:2px solid rgb(0, 100, 150);}")
        self.label_IDImg.move(580, 180)

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

        self.layout_identitys.addWidget(self.label_identity1, 0, Qt.AlignCenter | Qt.AlignTop)    
        
        self.layout_texts.addWidget(self.label_text1, 0, Qt.AlignCenter | Qt.AlignTop)      

        self.layout_faces.addLayout(self.layout_identitys)
        self.layout_faces.addLayout(self.layout_texts)
        
        self.layout_IC.addWidget(self.btn_IC)
        self.layout_IC.addWidget(self.label_ic, 0, Qt.AlignLeft)

        self.layout_face_IC.addLayout(self.layout_faces)
        self.layout_face_IC.addLayout(self.layout_IC)
        
        
        self.layout_show.addWidget(self.label_show_camera, 0, Qt.AlignCenter | Qt.AlignTop) 
        self.layout_show.addWidget(self.title_label)
               

        self.layout_main.addLayout(self.layout_button)
        self.layout_main.addLayout(self.layout_show)
        self.layout_main.addLayout(self.layout_face_IC)
        
        self.setLayout(self.layout_main)
        self.setGeometry(500, 200, 640*self.r_w, 480*self.r_h)
        self.use_palette()

    def use_palette(self): 
        self.setWindowTitle("门禁系统") 
        window_pale = QPalette() 
        window_pale.setBrush(self.backgroundRole(), QBrush(QPixmap("6.jpg"))) 
        self.setPalette(window_pale)

    def ID_detect_init(self):
        self.ID_detector = windll.LoadLibrary("termb.dll")
        ret = self.ID_detector.InitComm(1001)
        if ret==0:
            self.label_IDText.setText(u"设备连接失败")
        else:
            self.label_IDText.setText(u"请放卡")
            self.timer_ID.start(30)

    def ID_detect(self):
        flag = self.ID_detector.Authenticate()
        if flag == 1:
            print('Authenticate成功')
            read_flag = self.ID_detector.Read_Content(1)
            #print read_flag,"------read_flag---------"
            if read_flag==1:
                print('信息读取成功')
                path = os.getcwd()
                photo_path = path+"xp.wlt"
                self.ID_detector.GetBmpPhoto(photo_path)
                self.label_IDImg.setPixmap(QPixmap('zp.jpg'))
                portrait = cv2.imread('zp.jpg')
                portrait = cv2.cvtColor(portrait, cv2.COLOR_BGR2RGB)
                self.ID_ifo = np.array(self.detector.get_128d_features(portrait[40:100, 25:80]))
                self.ID_ready = True 
                self.label_IDText.setText(u"身份信息获取成功")

    def IC_card_detect(self):
        # try:
        #     self.flag_IC = True
        #     print('trying to open IC device')
        #     self.ser = serial.Serial('com4',9600)
        #     print('IC device opened')
        #     self.ser.write(bytes.fromhex('03 08 C1 20 02 00 00 17') )    #向串口输入ctrl+c
        #     self.ser.read(8)
        #     self.ser.timeout = 0.1
        #     print('write done')
        #     while self.flag_IC:
        #         data= str(binascii.b2a_hex(self.ser.read(12)))[2:-1]
        #         if data:
        #             print('读取到的卡号:', data[-10:-2])
        # except Exception as ex:
        #     self.ser.close()
        #     print('IC 刷卡设备访问出错: ' + str(ex))
        data= str(binascii.b2a_hex(self.ser.read(12)))[2:-1]
        if data[10:14]=='0400':
            self.label_ic.setText(data[10:-2])
            # print('读取到的卡号:', data[-10:-2])


    # 信号槽设置
    def slot_init(self):
        self.btn_open_cam.clicked.connect(self.btn_open_cam_click)
        self.btn_detection_face.clicked.connect(self.detect_face)
        self.btn_capture_face.clicked.connect(self.capture_face)
        self.btn_save_face.clicked.connect(self.save_face)
        self.btn_IC.clicked.connect(self.btn_IC_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.timer_serial.timeout.connect(self.IC_card_detect)
        self.timer_ID.timeout.connect(self.ID_detect)
        self.quit.clicked.connect(self.close)

    def btn_IC_click(self):
        if self.timer_serial.isActive() == False:
            try:
                self.ser = serial.Serial('com4',9600)
                print('IC device opened')
                self.ser.write(bytes.fromhex('03 08 C1 20 02 00 00 17') )    #向串口输入ctrl+c
                self.ser.read(8)
                self.ser.timeout = 0.01
                print('write done')
                self.timer_serial.start(30)
                self.btn_IC.setText(u'关闭IC功能')
            except Exception as ex:
                print('IC 设备访问出错： ' + str(ex))
        else:
            self.ser.close()
            self.timer_serial.stop()
            self.btn_IC.setText(u'打开IC功能')


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
            show = cv2.resize(self.image, (self.label_show_camera.width(), self.label_show_camera.height()))
            show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 这里指的是显示原图
            # opencv 读取图片的样式，不能通过Qlabel进行显示，需要转换为Qimage QImage(uchar * data, int width,
            # int height, Format format, QImageCleanupFunction cleanupFunction = 0, void *cleanupInfo = 0)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            self.label_show_camera.setPixmap(QPixmap.fromImage(showImage))
        else:
            ret_1, self.image_1 = self.cap.read()
            if ret_1:
                detect_image, informations = self.detector.face_detect_ACS(self.image_1)
                detect_image = cv2.resize(detect_image, (self.label_show_camera.width(), self.label_show_camera.height()))
                detect_image = QImage(detect_image.data, detect_image.shape[1], detect_image.shape[0], QImage.Format_RGB888)
                self.label_show_camera.setPixmap(QPixmap.fromImage(detect_image))
                i = 0

                if len(informations)==1 and self.ID_ready:
                    e = np.linalg.norm(self.ID_ifo - self.detector.face_ifo)
                    if e<0.5:
                        self.label_IDText.setText(u"身份验证成功！")

                    print("e = ", e)

                for infm in informations:
                    if i >= 1:
                        break
                    if infm[1]=='unknow':
                        continue

                    print((infm[3][0], infm[3][1], infm[3][2], infm[3][3]))

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
                while i<1:
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
            try:
                self.ser.close()
            except Exception as ex:
                print(str(ex))
            self.timer_serial.stop()
            self.timer_ID.stop()
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
    print('end')