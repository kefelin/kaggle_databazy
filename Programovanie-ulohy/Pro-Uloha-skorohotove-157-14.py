import tkinter
canvas = tkinter.Canvas(width=500,height=500,bg="white")
canvas.pack()

canvas.create_text(250,490,text="1. Okno je od a 2. je po",font = "Arial 10 bold")
x = 0
for i in range (3): #vytvorenie buttonov
    x += 150
    canvas.create_rectangle (140+x-225,360,210+x-225,290,fill="lime green",outline="black", width=2)
canvas.create_text (250,325,text = "↑"+ 8 * " " + "=" + 8 * " "+ "↓",font = "Arial 40 bold")

#canvas.create_rectangle(65,290,140,365,fill="white",outline="black",width=2)   #tu su len akoze hitboxy pre to klikanie aby sa lahsie dalo urcit :D (moze sa vymazať tieto 3 lajny)
#canvas.create_rectangle(215,290,285,365,fill="white",outline="black",width=2)
#canvas.create_rectangle(365,290,435,365,fill="white",outline="black",width=2)

tipnute_cislo = 0
global maxi, mini
def kliknutie (sur): #iba suradnice tam kde tie akoze butony aby fungovali
    x,y = sur.x, sur.y
    if x > 65 and x <140 and y > 290 and y < 365:
        mini = tipnute_cislo + 1
        print ("vacsie")
    if x > 215 and x <285 and y > 290 and y < 365:
        maxi = tipnute_cislo - 1
        print ("to cislo")
    if x > 365 and x <435 and y > 290 and y < 365:
        canvas.create_text (250,150,text="Uhadol som a cislo je:"+ str(tipnute_cislo),tags="text1")
        print ("mensie")

    tipnute_cislo = (mini + maxi) // 2
    canvas.delete("text1")
    canvas.create_text (250,100,text="Je toto tvoje cislo?" + "     "+ str(tipnute_cislo),font = "Arial 20 bold" ,tags = "text1")
    miniv = tipnute_cislo
    mini += 1
    maxi += 1
    
    
    
entry1 = tkinter.Entry()
entry2 = tkinter.Entry()
maxi,mini = 0, 0
def od_do ():  #zožen od teba min a max
    global maxi, mini
    od = entry1.get()
    po = entry2.get()
    maxi , mini = int(po), int(od)
    print ("od", od,",", "po", po, maxi,mini) 

canvas.bind("<Button-1>", kliknutie)

entry1.pack()
entry2.pack()

button1 = tkinter.Button(text = "Potvrď", command = od_do)
button1.pack()

canvas.mainloop()