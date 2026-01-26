import tkinter
def najdi_pary(vyraz):
    zasobnik = []
    pary = []
    for i in range(len(vyraz)):
        znak = vyraz[i]
        if znak == "(":
            zasobnik.append(i)
        elif znak == ")":
            if len(zasobnik) == 0:
                return False, []
            lava = zasobnik[len(zasobnik)-1]
            pary.append((lava, i))
            zasobnik = zasobnik[:-1]

    if len(zasobnik) != 0:
        return False, []
    return True, pary
farby = ["red", "blue", "green", "orange", "purple", "brown", "magenta"]
def zobraz_vyraz(vyraz):
    okno = tkinter.Canvas(width=500, height=200, bg = "white")
    platne, pary = najdi_pary(vyraz)
    okno.pack()
    text_id = []
    pozicie = []

    x = 10
    for i in range(len(vyraz)):
        znak = vyraz[i]
        id_znaku = okno.create_text(x, 50, text=znak, font=("Arial", 20), fill="black")
        text_id.append(id_znaku)
        pozicie.append(x)
        x += 20
    if platne:
        for idx in range(len(pary)):
            l, p = pary[idx]
            farba = farby[idx % len(farby)]
            okno.itemconfig(text_id[l], fill=farba)
            okno.itemconfig(text_id[p], fill=farba)
        okno.create_text(200, 80, text="Uzátvorkovanie je správne", font=("Arial", 14), fill="darkgreen")
    else:
        okno.create_text(200, 80, text="Uzátvorkovanie je nesprávne", font=("Arial", 14), fill="red")
    okno.mainloop()
vyraz = input("Zadaj výraz so zátvorkami: ")
zobraz_vyraz(vyraz)
