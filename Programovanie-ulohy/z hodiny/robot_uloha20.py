import tkinter
canvas = tkinter.Canvas(width=400,height=400)
canvas.pack()

korx, kory = 200,200
uhol = 0
farbapera = "black"
hrubkapera = 1
opakuj = 0
zoznam = []

def vykonaj (prikaz):
    global korx, kory, uhol, farbapera, hrubkapera, opakuj, zoznam
    zakladny = True
    casti = prikaz.split()
    if casti [0] == "ciara":
        dlzka = int (casti[1])
        if uhol == 0:
            x,y = korx, kory-dlzka
        elif uhol == 90:
            x,y = korx+dlzka,kory
        elif uhol == 180:
            x,y = korx, kory + dlzka
        elif uhol == 270:
            x,y = korx - dlzka, kory
        else:
            x,y = korx, kory
        canvas.create_line (korx, kory, x, y, fill=farbapera, width=hrubkapera)
        korx,kory = x , y


    elif casti[0] == "vlavo":
        uhol -= 90
        if uhol < 0:
            uhol += 360
    elif casti[0] == "vpravo":
        uhol += (uhol + 90) % 360
    elif casti [0] == "farbapera":
        farbapera =casti[1]
    elif casti [0] == "hrubkapera":
        hrubkapera = casti[1]
    elif casti[0] == "opakuj":
        opakuj = int(casti[1])
        zakladny = False
    elif casti [0] == "koniecopakuj":
        for i in range (opakuj-1):
            opakuj = 0
            for podprikaz in zoznam:
                vykonaj(podprikaz)
        zakladny = False
    if zakladny and opakuj:
        zoznam.append (prikaz)

def filerun ():
    subor = open(entry1.get())
    riadok = subor.readline()
    while riadok != "":
        vykonaj(riadok.strip())
entry1 = tkinter.Entry()
entry1.insert(0,"core.txt")
entry1.pack()
button1 =tkinter.Button(text="vykonaj prikazy so suboru", command = filerun)
button1.pack()

        
canvas.mainloop()