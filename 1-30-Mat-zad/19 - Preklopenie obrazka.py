import tkinter
canvas = tkinter.Canvas(width=800, height=600, background="white")
canvas.pack()
subor = open("preklopenie_obrazka.txt", "r")
prvy_riadok = subor.readline().strip().split()
rozmery = int(prvy_riadok[0]) * int(prvy_riadok[1])
strana = 50
obrazok = []
jednotky = 0
for riadok in subor:
    riadok = riadok.strip()  
    hodnoty = riadok.split()  
    obrazok.append(hodnoty)           
def prekresli():
    global jednotky
    canvas.delete("all")     
    for r in range(len(obrazok)):      
        for s in range(len(obrazok[r])):   
            if obrazok[r][s] == "1": 
                jednotky += 1
                x = s * strana
                y = r * strana
                canvas.create_rectangle(x, y, x + strana, y + strana, fill="black")
def preklop():
    for r in range(len(obrazok)):
        for s in range(len(obrazok[r])):
            if obrazok[r][s] == "1":
                obrazok[r][s] = "0"
            else:
                obrazok[r][s] = "1"
    prekresli()

prekresli()
print("pocet pixelov v scene je: ",rozmery, "  a pocet jednotiek je: ", jednotky)
button = tkinter.Button(text="Preklop", command=preklop)
button.pack()
canvas.mainloop()
