import tkinter
canvas = tkinter.Canvas(width=500, height=500, bg = "white")
canvas.pack()
widget = tkinter.Entry()
widget.pack()
smery = ["vpravo","vlavo","hore","dole"]
pozicia = []
x,y,x1,y1 = 250,250,250,250
smer,dlzka = "hore",0
def kresli ():
    global x,y,x1,y1, dlzka, smer
    vstup = widget.get()
    split_vstup = vstup.split()
    if split_vstup[0] == "ciara":
        dlzka = int(split_vstup[1])
        print("do predu o: ", str(dlzka)) 
        if smer == "hore":
            y1 -= dlzka
        elif smer == "vpravo":
            x1 += dlzka
        elif smer == "vlavo":
            x1 -= dlzka
        elif smer == "dole":
            y1 += dlzka
        canvas.create_line(x,y,x1,y1, fill = "black", width = 3)
    if vstup == smery[0]:
        print("do prava")
        if smer == "hore":
            smer = "vpravo"
        elif smer == "vpravo":
            smer = "dole"
        elif smer == "dole":
            smer = "vlavo"
        elif smer == "vlavo":
            smer = "hore"
    if vstup == smery[1]:
        print("do lava")
        if smer == "hore":
            smer = "vlavo"
        elif smer == "vlavo":
            smer = "dole"
        elif smer == "vpravo":
            smer = "hore"
        elif smer == "dole":
            smer = "vpravo"
    x, y = x1, y1        
Button = tkinter.Button(text="vykonaj", command= kresli)
Button.pack()
canvas.mainloop()