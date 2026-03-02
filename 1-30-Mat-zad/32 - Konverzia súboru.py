import tkinter
subor1 = open("ciernobiely_obrazok_1.txt","r",encoding="utf-8")
subor2 = open("ciernobiely_1_a_0.txt","w",encoding="utf-8")
prvy_riadok = subor1.readline()
prvy_riadok_split = prvy_riadok.strip().split()
riadky = subor1.readlines() 
sirka = int(prvy_riadok_split[0])
vyska = int(prvy_riadok_split[1])
canvas = tkinter.Canvas(width=sirka,height=vyska)
canvas.pack()
for riadok in riadky:
    subor2.write("\n")
    for farba in range(0, len(riadok.strip()), 2):
        hex_farba = riadok[farba:farba+2]
        tk_farba = int(hex_farba, 16)
        if tk_farba < 128:
            subor2.write(" " + "0")
        else:
            subor2.write(" " + "1")
subor1.close()  