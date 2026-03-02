import tkinter
subor1 = open("ciernobiely_obrazok_1.txt","r",encoding="utf-8")
prvy_riadok = subor1.readline()
prvy_riadok_split = prvy_riadok.strip().split()
riadky = subor1.readlines() 
sirka = int(prvy_riadok_split[0])
vyska = int(prvy_riadok_split[1])
canvas = tkinter.Canvas(width=sirka,height=vyska)
canvas.pack()
pocet = [0] * 256

for riadok in riadky:
    strip_riadok = riadok.strip()
    for i in range(0, len(strip_riadok),2):
        hex_farba = strip_riadok[i:i+2]
        cislo_farba = int(hex_farba, 16)
        pocet[cislo_farba] += 1
        
najviac = max(pocet)
mierka = 500 / najviac 
x = -250
for i in range(256):
    vyska_stlpca = pocet[i] * mierka
    canvas.create_line(x, 500, x, 500 - vyska_stlpca, width=2, fill="grey")
    x += 2
print(pocet)
canvas.mainloop()