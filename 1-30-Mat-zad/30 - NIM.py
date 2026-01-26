import tkinter, random
canvas = tkinter.Canvas(width=650, height=200,bg="white")
canvas.pack()
canvas.focus_set()
pocet = 15
hrac_cislo = 1
def zapalka(x, y):
    for i in range(pocet):
        x+=40
        canvas.create_line(x, y, x, y+100, width=5, fill='orange')
        canvas.create_oval(x-5, y-5, x+5, y+8, fill='brown', outline='brown')
x1 = 610
def stlacenie(event):
    global pocet,x1,hrac_cislo
    if event.keysym == "1":
        pocet -= 1
        canvas.delete("text")
        canvas.create_line(x1,200,x1,50, width=15, fill="white")
    elif event.keysym == "2":
        pocet -= 2
        canvas.delete("text")
        canvas.create_line(x1,200,x1,50, width=15, fill="white")
        canvas.create_line(x1 - 40,200,x1 - 40,50, width=15, fill="white")
        x1 -= 40
    elif event.keysym == "3":
        pocet -= 3
        canvas.delete("text")
        canvas.create_line(x1,200,x1,50, width=15, fill="white")
        canvas.create_line(x1 - 40,200,x1 - 40,50, width=15, fill="white")
        canvas.create_line(x1 - 80,200,x1 - 80,50, width=15, fill="white")
        x1 -= 80
    if hrac_cislo == 1:
        hrac_cislo = 2
    elif hrac_cislo == 2:
        hrac_cislo = 1
    x1 -=40
    text()
def text ():
    global pocet, hrac_cislo
    canvas.create_text(325,25 ,text="Počet zapaliek je: "+str(pocet), font="Arial 15 bold",tags="text")
    canvas.create_text(325,50 ,text="Na rade je hráč: "+str(hrac_cislo), font="Arial 15 bold",tags="text")
    if pocet <= 0:
        if hrac_cislo == 2:
            hrac_cislo = 1
        elif hrac_cislo == 1:
            hrac_cislo = 2
        canvas.delete("text")
        canvas.create_text(325,35 ,text="Vyhral hráč: " + str(hrac_cislo), font="Arial 15 bold",tags="text1")
canvas.bind("1",stlacenie)
canvas.bind("2",stlacenie)
canvas.bind("3",stlacenie)
zapalka(10,75)
text()
canvas.mainloop()