import tkinter

canvas = tkinter.Canvas(width=1920, height=1080, bg="black")
canvas.pack()

subor_x = open("x.txt", "r")
subor_y = open("y.txt", "r")

riadky_x = subor_x.readlines()
riadky_y = subor_y.readlines()

for i in range(len(riadky_x)):
    x = int(riadky_x[i].strip())
    y = int(riadky_y[i].strip())
    canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="white", outline="")
    canvas.update()
    canvas.after(2)

canvas.mainloop()