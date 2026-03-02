import tkinter
def vykresli_krizovku(x_start, y_start, velkost, vyplnena):
    subor = open("krizovka2-1.txt", "r", encoding="utf-8")
    tajnicka = subor.readline().strip()
    riadky_slov = subor.readlines()
    subor.close()
    y = y_start
    for i in range(len(tajnicka)):
        slovo = riadky_slov[i].strip()
        pismeno_tajnicky = tajnicka[i]
        pozicia_v_slove = slovo.find(pismeno_tajnicky)
        x_aktualne = x_start - (pozicia_v_slove * velkost)
        for j in range(len(slovo)):
            farba_pozadia = "white"
            if j == pozicia_v_slove:
                farba_pozadia = "grey"    
            canvas.create_rectangle(x_aktualne, y, x_aktualne + velkost, y + velkost, fill=farba_pozadia)
            if vyplnena == True:
                canvas.create_text(x_aktualne + velkost//2, y + velkost//2, text=slovo[j],font = "Arial 15 bold")
            x_aktualne = x_aktualne + velkost
        y = y + velkost

canvas = tkinter.Canvas(width=800, height=600, bg="white")
canvas.pack()
vykresli_krizovku(150, 50, 30, False)
vykresli_krizovku(500, 50, 30, True)
canvas.mainloop()