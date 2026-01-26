import tkinter
import random
canvas = tkinter.Canvas(width=700, height=800)
canvas.pack()

def lodicka(x, y):
    plachta = random.randint(-1, 2)
    canvas.create_line(x, y, x, y-25, x+10+plachta, y-10, x, y-5)
    canvas.create_polygon(x-20, y, x+20, y, x+10, y+8, x-10, y+8, x-10, y+8)

canvas.create_line(25,0,25,800,fill="black",width=2)

pozicie_y = []
for i in range (15):
    pozicie_y.append(60 + i*45)

pozicie_x = []
for i in range(15):
    pozicie_x.append(50)

ide = True
c_vyhercu = 0

while ide:
    canvas.delete("all")
    canvas.create_line(25,0,25,800,fill="black",width=2)
    canvas.create_line(675,0,675,800,fill="green",width=2)

    for i in range(15):
        pozicie_x[i] = pozicie_x[i] + random.randint(-2,8)
        lodicka(pozicie_x[i], pozicie_y[i])
        if pozicie_x[i] >= 675 and c_vyhercu == 0:
            c_vyhercu = i +1 
            ide = False

    canvas.update()
    canvas.after(50)

canvas.create_text(350, 400, text="Vyherca je: " + str(c_vyhercu))
canvas.mainloop()