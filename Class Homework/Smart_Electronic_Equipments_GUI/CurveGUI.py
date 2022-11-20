import time
import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
import tkinter as tk
import threading
import multiprocessing
from multiprocessing import shared_memory
from tkinter import scrolledtext, messagebox
from tkinter.constants import END, INSERT, DISABLED, NORMAL
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk 
from utils import Data

class Curve:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.geometry('800x600')
        self.root.title('曲线绘制')
        self.chose_node = [True, False, False, False]
        self.chose_param = [False, False, False]
        self.first_click = False

        self.db = Data()
        self.timer = threading.Timer(0.5, self.update_data)
        self.timer.start()
        self.timer.join()
        
        self.AddMenu()
        self.CreatWidgets()
        self.RefreshFigure()
    
    def update_data(self):
        self.db.load(name='running')
        # print(self.db.get_new())
        self.timer = threading.Timer(0.5, self.update_data)
        self.timer.start()
        
    def CreatWidgets(self):
        self.fig = Figure(figsize = (7, 7), dpi = 100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.get_tk_widget().pack(side = 'left', fill = 'y', expand=0)
        self.canvas._tkcanvas.pack(side = 'left', fill = 'y', expand=0)

        toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        toolbar.update()

    def AddMenu(self):
        self.menubar = tk.Menu(self.root)                  
        
        self.StartMenu = tk.Menu(self.menubar, tearoff = False)
        self.StartMenu.add_command(label = '重置', command = self.Refresh)
        self.StartMenu.add_command(label = '保存图片', command = self.save)
        self.StartMenu.add_separator()
        self.StartMenu.add_command(label = '退出', command = self.quit)
        self.menubar.add_cascade(label='开始(S)', menu = self.StartMenu)

        self.ViewMenu1 = tk.Menu(self.menubar, tearoff = False)
        self.ViewMenu1.add_command(label = '节点1', command = self.click_node_1)
        self.ViewMenu1.add_command(label = '节点2', command = self.click_node_2)
        self.ViewMenu1.add_command(label = '节点3', command = self.click_node_3)
        self.ViewMenu1.add_command(label = '节点4', command = self.click_node_4)
        self.menubar.add_cascade(label='选择节点', menu = self.ViewMenu1)

        self.ViewMenu2 = tk.Menu(self.menubar, tearoff = False)
        self.ViewMenu2.add_command(label = '温度', command = self.click_param_1)
        self.ViewMenu2.add_command(label = '湿度', command = self.click_param_2)
        self.ViewMenu2.add_command(label = '光强', command = self.click_param_3)
        self.menubar.add_cascade(label='选择参数', menu = self.ViewMenu2)

        self.root['menu'] = self.menubar

    def prepare_data(self):
        legends_all = ['1 T', '1 H', '1 L', '2 T', '2 H', '2 L', '3 T', '3 H', '3 L', '4 T', '4 H', '4 L']
        
        if self.chose_node[0]:
            d = self.db.get_node(1).values
            legends = legends_all[0:3]
            title = 'Node 1'
        elif self.chose_node[1]:
            d = self.db.get_node(2).values
            legends = legends_all[3:6]
            title = 'Node 2'
        elif self.chose_node[2]:
            d = self.db.get_node(3).values
            legends = legends_all[6:9]
            title = 'Node 3'
        elif self.chose_node[3]:
            d = self.db.get_node(4).values
            legends = legends_all[9:]
            title = 'Node 4'

        elif self.chose_param[0]:
            d = self.db.get_param('T').values
            legends = [legends_all[0], legends_all[3], legends_all[6], legends_all[9]]
            title = 'Temperature of 4 nodes'
        elif self.chose_param[1]:
            d = self.db.get_param('H').values
            legends = [legends_all[1], legends_all[4], legends_all[7], legends_all[10]]
            title = 'Humidity of 4 nodes'
        elif self.chose_param[2]:
            d = self.db.get_param('L').values
            legends = [legends_all[2], legends_all[5], legends_all[8], legends_all[11]]
            title = 'Lux of 4 nodes'
        return d, legends, title

    def RefreshFigure(self):
        
        d, legends, title = self.prepare_data()
        
        self.ax.clear()
        for i in range(len(legends)):
            self.ax.plot(d[:, 0], d[:, i+1], label = legends[i])
        self.ax.legend(loc='lower left')
        self.ax.set_xticks([])
        self.ax.set_title(title)
        self.canvas.draw()
        self.root.after(2, self.RefreshFigure)
        pass
    
    def save(self):
        time_now = time.strftime("%Y%m%d%H%M%S", time.localtime())
        self.fig.savefig('./figures/' + time_now + '.png')
        
    def quit(self):
        self.save()
        self.timer.cancel()
        self.root.destroy()
        self.root.quit()
    
    def Refresh(self):
        self.chose_node = [True, False, False, False]
        self.chose_param = [False, False, False]

    def click_node_1(self):
        self.chose_node = [True, False, False, False]
        self.chose_param = [False, False, False]
    
    def click_node_2(self):
        self.chose_node = [False, False, False, False]
        self.chose_node[1] = True
        self.chose_param = [False, False, False]

    def click_node_3(self):
        self.chose_node = [False, False, False, False]
        self.chose_node[2] = True
        self.chose_param = [False, False, False]

    def click_node_4(self):
        self.chose_node = [False, False, False, False]
        self.chose_node[3] = True
        self.chose_param = [False, False, False]

    def click_param_1(self):
        self.chose_param = [False, False, False]
        self.chose_param[0] = True
        self.chose_node = [False, False, False, False]

    def click_param_2(self):
        self.chose_param = [False, False, False]
        self.chose_param[1] = True
        self.chose_node = [False, False, False, False]

    def click_param_3(self):
        self.chose_param = [False, False, False]
        self.chose_param[2] = True
        self.chose_node = [False, False, False, False]