import tkinter
canvas = tkinter.Canvas(width=800, height=400)
canvas.pack()
subor = open("krizovka1-1.txt", "r", encoding="utf-8")
riadky = subor.readlines()
subor.close()
info = []
i = 0
while i < len(riadky):
    riadok = riadky[i].strip()
    if riadok != "":
        medzera = riadok.find(" ")
        cislo = int(riadok[:medzera])
        slovo = riadok[medzera+1:]
        info.append([cislo, slovo])
    i = i + 1
velkost = 30
x_stred = 200
y_start = 50
i = 0
while i < len(info):    # tu sa bude kreslit nevyplnena krizovka
    cislo = info[i][0]
    slovo = info[i][1]
    y = y_start + i * velkost
    x_zaciatok = x_stred - velkost * (cislo - 1)
    j = 0
    while j < len(slovo):
        x = x_zaciatok + j * velkost
        farba = "white"
        if j + 1 == cislo:
            farba = "lightgreen"
        canvas.create_rectangle(x, y, x + velkost, y + velkost, fill=farba)
        j = j + 1
    i = i + 1
x_stred2 = 600
i = 0
while i < len(info): # tu vyplnena krizovka
    cislo = info[i][0]
    slovo = info[i][1]
    y = y_start + i * velkost
    x_zaciatok = x_stred2 - velkost * (cislo - 1)
    j = 0
    while j < len(slovo):
        x = x_zaciatok + j * velkost
        farba = "white"
        if j + 1 == cislo:
            farba = "lightgreen"
        canvas.create_rectangle(x, y, x + velkost, y + velkost, fill=farba)
        canvas.create_text(x + velkost // 2, y + velkost // 2, text=slovo[j], 
                           font="Arial 14")
        j = j + 1
    i = i + 1
canvas.mainloop()