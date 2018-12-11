# 读一次卡号：01 08 A1 20 00 01 00 76
# 自动读卡号：03 08 C1 20 02 00 00 17 
# 关闭蜂鸣器：03 08 C2 20 00 00 00 16 
# 打开蜂鸣器：03 08 C2 20 01 00 00 17
import serial
import binascii
import time

t = serial.Serial('com4',9600)

t.write(bytes.fromhex('03 08 C1 20 02 00 00 17') )    #向串口输入ctrl+c
t.read(8)
t.timeout = 0.1
print('write done')
while True:
    data= str(binascii.b2a_hex(t.read(12)))[2:-1]
    if data:
        print('读取到的卡号:', data[-10:-2])


from ctypes import *
import os
dll = windll.LoadLibrary("termb.dll")

ret = dll.InitComm(1001)
if ret!=0:
    print('连接身份证成功，请放卡！')
else:
    exit()

while 1:
    flag = dll.Authenticate()
    if flag == 1:
        print('Authenticate成功')
        read_flag = dll.Read_Content(1)
        #print read_flag,"------read_flag---------"
        if read_flag==1:
            print('信息读取成功')
            path = os.getcwd()
            photo_path = path+"xp.wlt"
            dll.GetBmpPhoto(photo_path)