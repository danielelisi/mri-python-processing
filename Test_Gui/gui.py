from tkinter import *
import matplotlib as mp
mp.use("TkAgg")

# root = Tk()

# topFrame = Frame(root)
# topFrame.pack()
# bottomFrame = Frame(root)
# bottomFrame.pack(side=BOTTOM)

# button1 = Button(topFrame, text='Button 1', fg='red')
# button2 = Button(topFrame, text='Button 2', fg='blue')
# button3 = Button(topFrame, text='Button 3', fg='green')
# button4 = Button(bottomFrame, text='Button 4', fg='purple')

# button1.pack(side=LEFT)
# button2.pack(side=LEFT)
# button3.pack(side=LEFT)
# button4.pack(side=BOTTOM)

# root.mainloop()

#
# Part 3 - Fitting Widgets in The Layout
#

# root = Tk()

# one = Label(root, text='One', bg='red', fg='white')
# one.pack()

# two = Label(root, text='Two', bg='green', fg='black')
# two.pack(fill=X)

# three = Label(root, text='Three', bg='blue', fg='white')
# two.pack(side=LEFT, fill=Y)

# root.mainloop()

#
# Part 4 - Grid Layout
#

# root = Tk()

# label_1 = Label(root, text='Name')
# label_2 = Label(root, text='Password')
# entry_1 = Entry(root)
# entry_2 = Entry(root)

# label_1.grid(row=0)
# label_2.grid(row=1)

# entry_1.grid(row=0, column=1)
# entry_2.grid(row=1, column=1)

# root.mainloop()

#
# Part 5 - More on Grid Layout
#

# root = Tk()

# label_1 = Label(root, text='Name')
# label_2 = Label(root, text='Password')
# entry_1 = Entry(root)
# entry_2 = Entry(root)

# label_1.grid(row=0, sticky=E)     #Sticky doesn't use L,R,T,B but N,E,S,W
# label_2.grid(row=1, sticky=E)

# entry_1.grid(row=0, column=1)
# entry_2.grid(row=1, column=1)

# c = Checkbutton(root, text='Keep me logged in')
# c.grid(columnspan=2)

# root.mainloop()

#
# Part 6 - Binding Functions to Layouts
#

# root = Tk()

# def printName(event # one way to do an event):
#     print('Hello my name is Alex!')

# button_1 = Button(root, text='Print my name', command=printName # another way to do event)
# button_1.bind('<Button-1>', printName)
# button_1.pack()

# root.mainloop()p

#
# Part 7 - Mouse Click Events
#

# root = Tk()

# def leftClick(event):
#     print('Left')

# def middleClick(event):
#     print('Middle')

# def rightClick(event):
#     print('Right')

# frame = Frame(root, width=300, height=250)
# frame.bind('<Button-1>', leftClick)
# frame.bind('<Button-2>', middleClick)
# frame.bind('<Button-3>', rightClick)
# frame.pack()

# root.mainloop()

#
# Part 8 - Using Classes
#

class AlexButtons:

    def __init__(self, master):
        frame = Frame(master)
        frame.pack()

        self.printButton = Button(frame, text='Print Message', command=self.printMessage)
        self.printButton.pack(side=LEFT)

        self.quitButton = Button(frame, text='Quit', command=frame.quit)
        self.quitButton.pack(side=LEFT)

    def printMessage(self):
        print('Wow, this actually works!')

root = Tk()

b = AlexButtons(root)

root.mainloop()

