import tkinter as tk
import random
from tkinter import messagebox


class GUI:
    
    def __init__(self):
        self.window = tk.Tk()
        self.window.title('抽取题目')
        self.window.geometry('600x600+600+225')
        self.title = ['1-数据预处理', '1-图像处理', '1-文本处理',
                      '1-热度预测', '1-有效性验证', '1-假新闻过滤模型',
                      '1-群策群力', '1-模型格式转换', '1-模型部署',
                      '2-序列标注', '2-日志记录', '2-模型量化',
                      '3-无监督算法应用', '3-预训练模型应用', '4-账号风险评估',
                      '4-文本表示','4-数据处理']
        random.shuffle(self.title)  # 打乱题目列表
        self.index = 0  # 初始化索引
        self.isrolling = False  # 添加一个状态标志
        self.roll_id = None  # 添加一个计时器ID变量
        self.interface()
    
    def interface(self):
        title_window = tk.Label(self.window, text='题目总结', justify='center')
        title_window.pack()
        content = tk.Message(self.window, text=self.title, justify='center', width=600, bg='skyblue', font=6)
        content.pack()
        self.txt = tk.Label(self.window, text='', bg='#DAD0FA', width=68, pady=40, font=6)
        self.txt.pack(pady=30)
        btn = tk.Button(self.window, text='开始/停止滚动', width=25, height=10, font=8, bg='#D7E3BC',
                        command=self.toggle_rolling)
        btn.pack(pady=50)
    
    def toggle_rolling(self):
        if self.isrolling:  # 如果当前是滚动状态，则停止滚动并确定当前题目
            self.isrolling = False
            self.window.after_cancel(self.roll_id)  # 取消计时器
            self.roll_id = None
            self.confirm_current_title()
        else:  # 如果当前是停止状态，则开始滚动
            if not self.title:  # 如果所有题目都已确定，则不执行滚动
                messagebox.showinfo('提示', '所有题目已确定')
                return
            self.isrolling = True
            self.roll_titles()
    
    def roll_titles(self):
        if not self.isrolling:  # 如果不是滚动状态，则不执行
            return
        self.txt.config(text=self.title[self.index])
        self.index = (self.index + 1) % len(self.title)  # 更新索引，循环显示
        self.roll_id = self.window.after(100, self.roll_titles)  # 设置计时器
    
    def confirm_current_title(self):
        # 移除当前显示的题目
        current_title = self.title.pop(self.index - 1)
        print(f"已确定题目：{current_title}")
        # 如果列表为空，则禁用按钮
        if not self.title:
            self.txt.config(text="")
            messagebox.showinfo('提示', '所有题目已确定')
            self.toggle_rolling.config(state='disabled')


g = GUI()
g.window.mainloop()
