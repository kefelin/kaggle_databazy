import tkinter

canvas = tkinter.Canvas(width=400, height=200, bg="black")
canvas.pack()

rychl = 5

def nacitaj_zastavky():
    f = open("zastavky.txt", "r", encoding="utf-8")
    riadky = []
    for r in f:
        r = r.strip()
        if r != "":
            riadky.append(r)
    f.close()
    return riadky

def start(event):
    global index, x, text
    if index >= len(zastavky):
        index = 0
    text = zastavky[index]
    index = index + 1
    x = 0

def animuj():
    global x
    canvas.delete("all")
    canvas.create_text(x, 200//2, text=text, font="Arial 15 bold", fill="red")
    x = x + rychl
    if x > 400:
        x = 0
    canvas.after(25, animuj)

zastavky = nacitaj_zastavky()
index = 0
x = 0
text = ""

canvas.bind("<Button-1>", start)

animuj()
canvas.mainloop()
