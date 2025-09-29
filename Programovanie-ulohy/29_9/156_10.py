import tkinter
canvas = tkinter.Canvas(width=500, height=300)
canvas.pack()


zapalky_pocet = 0
hrac = 1
tahy = []

def zapalky():
    global zapalky_pocet, hrac, tahy
    canvas.delete("zap")
    x = 0
    zapalky_pocet = int(entry1.get())
    hrac = 1
    tahy = []
    text.config(text="Hráč 1 je na ťahu")
    for i in range(zapalky_pocet):
        x += 40
        canvas.create_line(25 + x, 280, 25 + x, 100, fill="orange", width=8, tags="zap")
        canvas.create_oval(10 + x, 100, 40 + x, 50, fill="red", outline="black", tags="zap")

def zobrat1():
    zobrat(1)

def zobrat2():
    zobrat(2)

def zobrat3():
    zobrat(3)

def zobrat(pocet):
    global zapalky_pocet, hrac
    if zapalky_pocet == 0:
        return
    if pocet > zapalky_pocet:
        pocet = zapalky_pocet
    zapalky_pocet = zapalky_pocet - pocet
    tahy.append("Hráč " + str(hrac) + " zobral " + str(pocet) + " zápalky")
    canvas.delete("zap")
    x = 0
    for i in range(zapalky_pocet):
        x += 40
        canvas.create_line(25 + x, 280, 25 + x, 100, fill="orange", width=8, tags="zap")
        canvas.create_oval(10 + x, 100, 40 + x, 50, fill="red", outline="black", tags="zap")
    if zapalky_pocet == 0:
        text.config(text="Hráč " + str(hrac) + " prehral! Vyhráva hráč " + str(3 - hrac))
    else:
        hrac = 3 - hrac
        text.config(text="Hráč " + str(hrac) + " je na ťahu")

def repriza():
    canvas.delete("zap")
    x = 0
    for i in range(int(entry1.get())):
        x += 40
        canvas.create_line(25 + x, 280, 25 + x, 100, fill="gray", width=8, tags="zap")
        canvas.create_oval(10 + x, 100, 40 + x, 50, fill="darkred", outline="black", tags="zap")
    vystup = "Repríza:\n"
    for i in range(len(tahy)):
        vystup = vystup + tahy[i] + "\n"
    text.config(text=vystup)

entry1 = tkinter.Entry()
entry1.pack()

button1 = tkinter.Button(text="zapni", command=zapalky)
button1.pack()

text = tkinter.Label(text="Zadaj počet zápaliek a klikni na 'zapni'")
text.pack()

button_z1 = tkinter.Button(text="Zobrať 1", command=zobrat1)
button_z1.pack()

button_z2 = tkinter.Button(text="Zobrať 2", command=zobrat2)
button_z2.pack()

button_z3 = tkinter.Button(text="Zobrať 3", command=zobrat3)
button_z3.pack()

button_znova = tkinter.Button(text="Repríza", command=repriza)
button_znova.pack()

canvas.mainloop()

#Priznavam u tohto som si zacal pomahat copilotom pri odoberani a pri tych labeloch 
