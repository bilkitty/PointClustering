#!/usr/bin/python3
"""
ZetCode Tkinter tutorial

The example draws lines on the Canvas.

Author: Jan Bodnar
Last modified: April 2019
Website: www.zetcode.com
"""

#from tkinter import Tk, Canvas, Frame, BOTH
from tkinter import *


def clicked(lable):
    lable.configure(text="Drawing submitted!")


class Example(Frame):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.master.title("Lines")
        self.pack(fill=BOTH, expand=1)

    def create_lines(self):
        canvas = Canvas(self)
        canvas.create_line(15, 25, 200, 25)
        canvas.create_line(300, 35, 300, 200, dash=(4, 2))
        canvas.create_line(55, 85, 155, 85, 105, 180, 55, 85)

        canvas.pack(fill=BOTH, expand=1)


def main():
    window = Tk()
    ex = Example()

    window.title("Draw a spline")
    window.geometry("400x250+300+300")

    lbl = Label(window, text="Window")
    button = Button(window, text="Submit")

    window.mainloop()


if __name__ == '__main__':
    main()

