import tkinter
sirka = 400
vyska = 400
krok = 1   
smer = "hore"
had = [(200, 200)]
canvas = tkinter.Tk()
canvas.title("Háďa")
plocha = tkinter.Canvas(canvas, width=sirka, height=vyska, bg="white")
plocha.pack()
def vykresli_hada():
    plocha.delete("vsetko")
    suradnice = []
    for bod in had:
        suradnice.append(bod[0])
        suradnice.append(bod[1])
    if len(suradnice) >= 4:
        plocha.create_line(suradnice, fill="black", width=2)
def pohni_sa():
    global had
    hlavicka_x, hlavicka_y = had[-1]
    if smer == "hore":
        hlavicka_y = hlavicka_y - krok
    elif smer == "dole":
        hlavicka_y = hlavicka_y + krok
    elif smer == "vlavo":
        hlavicka_x = hlavicka_x - krok
    elif smer == "vpravo":
        hlavicka_x = hlavicka_x + krok
    nova_hlavicka = (hlavicka_x, hlavicka_y)
    had.append(nova_hlavicka)
    if narazil_do_seba() or narazil_do_steny():
        print("Koniec hry!")
        return
    vykresli_hada()
    canvas.after(10, pohni_sa)
def narazil_do_seba():
    hlavicka = had[-1]
    return hlavicka in had[:-1]
def narazil_do_steny():
    hlavicka_x, hlavicka_y = had[-1]
    if hlavicka_x < 0 or hlavicka_x > sirka or hlavicka_y < 0 or hlavicka_y > vyska:
        return True
    return False
def zmen_smer(udalost):
    global smer
    if udalost.keysym == "Up":
        smer = "hore"
    elif udalost.keysym == "Down":
        smer = "dole"
    elif udalost.keysym == "Left":
        smer = "vlavo"
    elif udalost.keysym == "Right":
        smer = "vpravo"
canvas.bind("<Key>", zmen_smer)
vykresli_hada()
canvas.after(50, pohni_sa)
canvas.mainloop()
