import os
import numpy as np
import tkinter as tk
import threading
import multiprocessing
from tkinter import scrolledtext, messagebox
from tkinter.constants import END, INSERT, DISABLED, NORMAL
from CurveGUI import Curve
from utils import Data
import serial
from time import sleep


helpstr = '''
作者：赵泓珏
时间：2021.10.28
邮箱：zhaohongjue123@163.com
使用请标明出处

本项目为2020-2021学年浙江大学控制科学与工程学院课程“智能电子设备开发”课程作业：分布式环境监测系统。题目要求如下：

1. 分为1个主节点，网关 4 个子节点；
2. 传感器模块通过接插件连接子节点板，子节点完成不同位置的温湿度光照的信号采集；
3. 主节点和子节点采用LORA通信，主节点通过232或WIFI和电脑手机相连。

本项目为分布式环境监测系统上位机的GUI。在本项目中我共实现了如下功能：
1. 实时显示四个节点的温度、湿度和光照强度；
2. 保存已检测到数据，保存绘制出的图像；
3. 校准节点温度值。

具体使用说明如下：
1. 主界面：
    1. 开始：
        1. 刷新：清除目前已存储的数据值，并将已测得数据以csv格式保存到指定目录，开始新一轮的测量；
        2. 保存：将已测得的数据保存到指定目录以csv格式保存到指定目录，但是不清除以存储数据值；
        3. 校准节点：手工设定目前的温度、湿度及光照，并对整个数据库进行更新；
        4. 快速校准：以当前四个节点测得温度、湿度、光照的均值作为校准值进行更新；
        5. 退出：保存数据，退出GUI
2. 查看-绘制曲线：
    1. 重置：显示节点1相关信息；
    2. 保存图像：将当前图像保存到指定目录；
    3. 退出：保存目前图像并退出。
    4. 选择节点/参数：绘制对应节点/参数的相关曲线。
3. 帮助：
    1. 文档：显示帮助文档（就是你在看得这个）
    2. 关于：显示开发者信息（也就是我的信息hhhhh）

本人水平程序设计水平有限，计算机底层知识了解不足，时间紧迫，实现过程中难免有所纰漏，还望指正。
'''

db = Data()
db.save(name='running')

# 实时获取串口数据
def get_ser_data(): 
    try:
        ser = serial.Serial('COM5', 9600, bytesize=8, parity='N', stopbits=1, timeout=1)
    
        receive_msg = np.zeros(5, dtype=int)     #本次接受的信息
        point1 = np.zeros(3)                #各个节点接受到的信息之和
        point2 = np.zeros(3)
        point3 = np.zeros(3)
        point4 = np.zeros(3)
        allpoints = np.zeros(12)   #所有节点一轮接受后的平均值
        modified = np.zeros(4)                #各个节点本轮接受的次数
        last_modified=0                     #上次接收的节点ID
        flag = False
        while True:
            head_msg = ser.read().hex()
            # print('head_msg', head_msg)
            sleep(0.01)
            if head_msg == 'af':            #如果帧头正确，则接受数据
                print('head_msg', head_msg)
                for i in range(5):
                    rmsg = ser.read().hex()
                    receive_msg[i] = int(rmsg, 16)
                    print('rmsg', rmsg, 'receive[i]', receive_msg[i])
                    sleep(0.01)
                
                if receive_msg[0] != last_modified and modified[0] != 0 and modified[1] != 0 and modified[2] != 0 and modified[3] != 0:
                    allpoints[0:3] = point1 / modified[0]
                    allpoints[3:6] = point2 / modified[1]
                    allpoints[6:9] = point3 / modified[2]
                    allpoints[9:] = point4 / modified[3]
                    flag = True
                    point1, point2, point3, point4, modified = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(4)
                    
                #判断ID，并将对应节点的值进行累加
                if receive_msg[0] == 2:
                    point1[1] = point1[1]+receive_msg[1]
                    point1[0] = point1[0]+receive_msg[2]
                    point1[2] = point1[2]+receive_msg[3]*(2**8)+receive_msg[4]
                    modified[0] = modified[0]+1
                    
                if receive_msg[0] == 3:
                    point2[1] = point2[1]+receive_msg[1]
                    point2[0] = point2[0]+receive_msg[2]
                    point2[2] = point2[2]+receive_msg[3]*(2**8)+receive_msg[4]
                    modified[1] = modified[1]+1
                    
                if receive_msg[0] == 4:
                    point3[1] = point3[1]+receive_msg[1]
                    point3[0] = point3[0]+receive_msg[2]
                    point3[2] = point3[2]+receive_msg[3]*(2**8)+receive_msg[4]
                    modified[2] = modified[2]+1
                    
                if receive_msg[0] == 5:
                    point4[1] = point4[1]+receive_msg[1]
                    point4[0] = point4[0]+receive_msg[2]
                    point4[2] = point4[2]+receive_msg[3]*(2**8)+receive_msg[4]
                    modified[3] = modified[3]+1
                
                last_modified = receive_msg[0]
            if flag:
                # print(allpoints)
                db.store(allpoints)
                print(db.df.values)
                db.save(name='running')
                flag = False
    except:
        messagebox.showerror('错误信息','未接入设备！')

# 帮助文档界面
def _HelpDoc():
    DocGUI = tk.Tk()
    DocGUI.title('帮助文档')
    DocGUI.geometry("500x500")
    doc = scrolledtext.ScrolledText(DocGUI, width = 68, height = 38, relief = 'solid')
    doc.grid(column=0, row=0)
    doc.config(state = NORMAL)
    doc.delete(1.0, END)
    doc.insert(INSERT, helpstr)
    doc.config(state = DISABLED)
    DocGUI.mainloop()

# 开发者信息界面
def _info():
    str = '智能电子设备开发-分布式检测系统GUI\n开发者：赵泓珏\n开发时间：2021.10.26'
    messagebox.showinfo("关于", str)

# 数据保存
def _SaveData():
    db.save()

#节点校准
def _Setting():
    SetGUI = tk.Tk()
    SetGUI.geometry('180x100')
    SetGUI.title('校准节点')
    tk.Label(SetGUI, text="设定温度值：").grid(column=0, row=0)
    tk.Label(SetGUI, text="设定湿度值：").grid(column=0, row=1)
    tk.Label(SetGUI, text="设定光照值：").grid(column=0, row=2)
    txt1 = tk.Entry(SetGUI, width=10)
    txt1.grid(column=1, row=0)
    txt2 = tk.Entry(SetGUI, width=10)
    txt2.grid(column=1, row=1)
    txt3 = tk.Entry(SetGUI, width=10)
    txt3.grid(column=1, row=2)
    def clicked1():
        txtlist = [txt1.get(), txt2.get(), txt3.get()]
        set_value = []
        for i in range(len(txtlist)):
            try:
                set_value.append(float(txtlist[i]))
            except:
                set_value.append(0)
        bias = db.get_new()[1:] - np.asarray(set_value*4)
        db.set_bias(np.asarray(bias))
        db.update()
        db.save(name='running')
        messagebox.showinfo('提示信息', '校准成功！')
        SetGUI.destroy()
        SetGUI.quit()

    def clicked2():
        SetGUI.destroy()
        SetGUI.quit()
    btn1 = tk.Button(SetGUI, text='确定',command=clicked1)
    btn1.grid(column=0, row=3)
    btn2 = tk.Button(SetGUI, text='退出',command=clicked2)
    btn2.grid(column=1, row=3)

    SetGUI.mainloop()

# 节点快速校准
def _Refresh():
    db.clear()

# 曲线绘制
def CreateCurve():
    curve = Curve()
    curve.root.mainloop()

# 快速校准
def _QuickSetting():
    data = db.get_new()
    set_value = [np.mean([data[i], data[i+3], data[i+6], data[i+9]]) for i in range(1, 4)]
    db.set_bias(data[1:] - np.asarray(set_value*4))
    db.update()
    db.save(name='running')
    messagebox.showinfo('提示','快速更新成功！')

# 测试函数
def get_data():
    global db
    data = np.ones(12) * 22
    data = data + np.random.randint(low = 0, high = 10, size = (12,)) * 3
    db.store(data)
    db.save(name='running')
    timer0 = threading.Timer(2, get_data)
    timer0.start()

thread0 = threading.Thread(target=get_ser_data)
thread0.daemon = True

class MainGUI:
    def __init__(self):
        # 窗口设置
        self.Width = 320            # 窗口宽度
        self.Height = 200           # 窗口高度

        self.root = tk.Tk()         
        self.root.geometry(str(self.Width) + 'x' + str(self.Height))
        self.root.title('分布式环境监测系统')
        self.root.iconbitmap('Icon.ico')

        self.AddContent()
        self.AddMenu()

        thread0.start()
        # thread0.join()

        self.RefreshTxt()
        self.root.mainloop()

    # 界面构建
    def AddContent(self):
        tk.Label(self.root, text='节点1', font=('黑体', 20)).grid(column=0, row = 0)
        self.txt1 = scrolledtext.ScrolledText(self.root, width = 20, height = 3)
        self.txt1.config(state = DISABLED)
        self.txt1.grid(column=0, row=1)

        tk.Label(self.root, text='节点2', font=('黑体', 20)).grid(column=1, row = 0)
        self.txt2 = scrolledtext.ScrolledText(self.root, width = 20, height = 3)
        self.txt2.config(state = DISABLED)
        self.txt2.grid(column=1, row=1)

        tk.Label(self.root, text='节点3', font=('黑体', 20)).grid(column=0, row = 2)
        self.txt3 = scrolledtext.ScrolledText(self.root, width = 20, height = 3)
        self.txt3.config(state = DISABLED)
        self.txt3.grid(column=0, row=3)

        tk.Label(self.root, text='节点4', font=('黑体', 20)).grid(column=1, row = 2)
        self.txt4 = scrolledtext.ScrolledText(self.root, width = 20, height = 3)
        self.txt4.config(state = DISABLED)
        self.txt4.grid(column=1, row=3)

    # 刷新内容
    def RefreshTxt(self):
        global db
        self.txt1.config(state = NORMAL)
        self.txt1.delete(1.0, END)
        str1 = '温度(℃)：{:.1f}\n湿度(%)：{:.1f}\n光照(lx)：{:.1f}'.format(db.get_new()[1], db.get_new()[2], db.get_new()[3])
        self.txt1.insert(INSERT, str1)
        self.txt1.config(state = DISABLED)

        self.txt2.config(state = NORMAL)
        self.txt2.delete(1.0, END)
        str2 = '温度(℃)：{:.1f}\n湿度(%)：{:.1f}\n光照(lx)：{:.1f}'.format(db.get_new()[4], db.get_new()[5], db.get_new()[6])
        self.txt2.insert(INSERT, str2)
        self.txt2.config(state = DISABLED)

        self.txt3.config(state = NORMAL)
        self.txt3.delete(1.0, END)
        str3 = '温度(℃)：{:.1f}\n湿度(%)：{:.1f}\n光照(lx)：{:.1f}'.format(db.get_new()[7], db.get_new()[8], db.get_new()[9])
        self.txt3.insert(INSERT, str3)
        self.txt3.config(state = DISABLED)

        self.txt4.config(state = NORMAL)
        self.txt4.delete(1.0, END)
        str4 = '温度(℃)：{:.1f}\n湿度(%)：{:.1f}\n光照(lx)：{:.1f}'.format(db.get_new()[10], db.get_new()[11], db.get_new()[12])
        self.txt4.insert(INSERT, str4)
        self.txt4.config(state = DISABLED)

        self.root.after(1000, self.RefreshTxt)

    # 菜单栏设置
    def AddMenu(self):
        self.menubar = tk.Menu(self.root)                  
        
        self.StartMenu = tk.Menu(self.menubar, tearoff = False)
        self.StartMenu.add_command(label = '刷新', command = self.Refresh)
        self.StartMenu.add_command(label = '保存数据', command = self.SaveData)
        self.StartMenu.add_separator()
        self.StartMenu.add_command(label = '校准节点', command = self.setting)
        self.StartMenu.add_command(label = '快速校准', command = self.quicksetting)
        self.StartMenu.add_separator()
        self.StartMenu.add_command(label = '退出', command = self.quit)
        self.menubar.add_cascade(label='开始(S)', menu = self.StartMenu)

        self.ViewMenu = tk.Menu(self.menubar, tearoff = False)
        self.ViewMenu.add_command(label = '绘制曲线', command = self.Curve)
        self.menubar.add_cascade(label='查看(V)', menu = self.ViewMenu)

        self.HelpMenu = tk.Menu(self.menubar, tearoff = False)
        self.HelpMenu.add_command(label = '文档', command = self.HelpDoc)
        self.HelpMenu.add_command(label = '关于', command = self.info)
        self.menubar.add_cascade(label='帮助(H)', menu = self.HelpMenu)

        self.root['menu'] = self.menubar
    
    # 帮助文档
    def HelpDoc(self):
        thread_HelpDoc = threading.Thread(target=_HelpDoc, name='HelpDoc')
        thread_HelpDoc.daemon = True
        thread_HelpDoc.start()

    # 开发者信息
    def info(self):
        thread_info = threading.Thread(target=_info, name='info')
        thread_info.daemon = True
        thread_info.start()

    # 刷新
    def Refresh(self):
        thread_refresh = threading.Thread(target=_Refresh, name = 'Refresh')
        thread_refresh.daemon = True
        thread_refresh.start()
        thread_refresh.join()

        messagebox.showinfo("提示信息", '刷新成功！数据已保存')
    
    # 绘制曲线
    def Curve(self):
        process_curve = multiprocessing.Process(target=CreateCurve, name = 'Curve')
        process_curve.daemon = True
        process_curve.start()

    # 保存数据
    def SaveData(self):
        thread_savedata = threading.Thread(target=_SaveData, name = 'SaveData')
        thread_savedata.daemon = True
        thread_savedata.start()

        messagebox.showinfo("提示信息", '保存成功！')

    # 退出
    def quit(self):
        db.save()
        os.remove('./data/running.csv')
        self.root.quit()
    
    # 节点校准
    def setting(self):
        thread_setting = threading.Thread(target=_Setting, name = 'setting')
        thread_setting.daemon = True
        thread_setting.start()

    # 快速校准
    def quicksetting(self):
        thread_quick = threading.Thread(target=_QuickSetting)
        thread_quick.daemon = True
        thread_quick.start()
    
if __name__ == '__main__':
    MainGUI()