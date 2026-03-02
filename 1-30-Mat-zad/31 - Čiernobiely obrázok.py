import tkinter
subor1 = open("ciernobiely_obrazok_1.txt","r",encoding="utf-8")
prvy_riadok = subor1.readline()
prvy_riadok_split = prvy_riadok.strip().split()
riadky = subor1.readlines() 

sirka = int(prvy_riadok_split[0])
vyska = int(prvy_riadok_split[1])

canvas = tkinter.Canvas(width=sirka,height=vyska)
canvas.pack()
y = 0
for riadok in riadky:
    x = 0
    for farba in range(0, len(riadok.strip()), 2):
        hex_farba = riadok[farba:farba+2]
        tk_farba = f"#{hex_farba}{hex_farba}{hex_farba}"
        canvas.create_rectangle(x,y,x+1,y+1,fill=tk_farba, outline=tk_farba)
        x += 1
    y += 1
    
def def_button():
    y1 = 0
    for riadok in riadky:
        x1 = 0
        for farba in range(0, len(riadok.strip()), 2):
            hex_farba = riadok[farba:farba+2]
            tk_farba = int(hex_farba, 16)
            if tk_farba < 128:
                canvas.create_rectangle(x1,y1,x1+1,y1+1,fill="black", outline="black")
            else:
                canvas.create_rectangle(x1,y1,x1+1,y1+1,fill="white", outline="white")
            x1 += 1
        y1 += 1
    print("na cierno bielo")
        
button = tkinter.Button(text="Na ČB", command=def_button)
button.pack()

canvas.mainloop()