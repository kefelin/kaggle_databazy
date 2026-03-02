import tkinter
subor = open("komprimovany_obrazok_1.txt", "r", encoding="utf-8")
riadok1 = subor.readline()
rozmery = riadok1.strip().split()
sirka = int(rozmery[0])
vyska = int(rozmery[1])
canvas = tkinter.Canvas(width=sirka, height=vyska + 40, bg="white")
canvas.pack()

def vykresli(styl):
    canvas.delete("all")
    subor_vystup = open("komprimovany_obrazok_1.txt", "r", encoding="utf-8")
    subor_vystup.readline()
    y = 0
    for riadok in subor_vystup:
        cisla = riadok.strip().split()
        x = 0
        farba_kod = 0
        for pocet in cisla:
            p = int(pocet)
            if styl == "normal":
                if farba_kod == 0:
                    farba = "black"
                else:
                    farba = "white"
            else:
                if farba_kod == 0:
                    farba = "white"
                else:
                    farba = "black"
            if p > 0:
                canvas.create_line(x, y, x + p, y, fill=farba)
            x = x + p
            if farba_kod == 0:
                farba_kod = 1
            else:
                farba_kod = 0
        y = y + 1
    subor_vystup.close()

def negativ():
    vykresli("negativ")
button = tkinter.Button(text="negativ", command=negativ)
button.pack()
vykresli("normal")
canvas.mainloop()
subor.close()