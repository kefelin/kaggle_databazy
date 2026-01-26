import tkinter
import random
canvas = tkinter.Canvas(width=500, height=500,bg = "lightblue")
canvas.pack()
x,y = 0,0
def u_k (event):
    farba = random.choice(["green","light green","lime green","dark green","yellow green","pale green","sea green","forest green"])
    body = []
    x = 0
    y = random.randint(250,450)
    vrchol = random.randint(100,400)
    smer = random.choice(["kopec","udolie"])
    for i in range(51):
        x = i * 10
        if smer == "kopec":
            if x > vrchol:
                y += random.randint(0,3)
            else:
                y -= random.randint(0,3)
        else:
            if x > vrchol:
                y -= random.randint(0,6)
            else:
                y += random.randint(0,6)
        body.append((x,y))
    body.append((500,500))
    body.append((0,500))
    body_funkcne = []
    for (x, y) in body:
        body_funkcne.append(x)
        body_funkcne.append(y)
    print(body_funkcne)
    canvas.create_polygon(body_funkcne, fill=farba)
    print(body)
canvas.bind_all("<space>",u_k)
canvas.mainloop()