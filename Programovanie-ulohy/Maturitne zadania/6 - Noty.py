import tkinter
canvas = tkinter.Canvas(width=800, height=400, bg="white")
canvas.pack()

def kresli_osnovu(y):
    for i in range(5):
        canvas.create_line(20, y + i*10, 780, y + i*10)

f = open("noty.txt", "r", encoding="utf-8")
noty = f.read().strip()
f.close()

x = 40
riadok = 0
max_sirka = 760
krok = 20

kresli_osnovu(50)

for nota in noty:
    y = 50 + riadok*100 + 40
    if nota == "c":
        y = y
    if nota == "d":
        y = y - 5
    if nota == "e":
        y = y - 10
    if nota == "f":
        y = y - 15
    if nota == "g":
        y = y - 20
    if nota == "a":
        y = y - 25
    if nota == "h":
        y = y - 30

    canvas.create_oval(x-5, y-5, x+5, y+5, fill="black")

    x = x + krok
    if x > max_sirka:
        riadok = riadok + 1
        kresli_osnovu(50 + riadok*100)
        x = 40

canvas.mainloop()
