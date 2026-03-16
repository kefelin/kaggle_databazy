import tkinter
import vlajka

canvas = tkinter.Canvas(width=600, height=250,bg="lightblue")
canvas.pack()

def nacitaj_krajiny(subor):
    krajiny = []
    x = 100
    for riadok in subor:
        casti = riadok.strip().split(" ")
        nazov = casti[0]
        rozloha = int(casti[1])
        populacia = int(casti[2])
        zvislo = True if casti[3] == "True" else False
        farby = casti[4:7]
        plat = int(casti[7])
        vl = vlajka.Vlajka(x, 120, 120, 80, zvislo, farby)
        vl.povodna_sirka = 120
        vl.povodna_vyska = 80
        krajiny.append([nazov, rozloha, populacia, plat, vl])
        x += 180
    return krajiny

def reset_vlajok():
    for krajina in krajiny:
        vl = krajina[4]
        vl.sirka = vl.povodna_sirka
        vl.vyska = vl.povodna_vyska

def prekresli():
    canvas.delete("all")
    for krajina in krajiny:
        krajina[4].kresli(canvas)

def podla_populacie():
    reset_vlajok()
    max_pop = max(k[2] for k in krajiny)
    for krajina in krajiny:
        pomer = krajina[2] / max_pop
        krajina[4].zoom(pomer)
    prekresli()

def podla_rozlohy():
    reset_vlajok()
    max_roz = max(k[1] for k in krajiny)
    for krajina in krajiny:
        pomer = krajina[1] / max_roz
        krajina[4].zoom(pomer)
    prekresli()

def podla_platu():
    reset_vlajok()
    max_plat = max(k[3] for k in krajiny)
    for krajina in krajiny:
        pomer = krajina[3] / max_plat
        krajina[4].zoom(pomer)
    prekresli()

with open("udaje.txt", "r", encoding="utf-8") as f:
    krajiny = nacitaj_krajiny(f)

prekresli()

tkinter.Button(text="Podľa populácie", command=podla_populacie).pack()
tkinter.Button(text="Podľa rozlohy", command=podla_rozlohy).pack()
tkinter.Button(text="Podľa priemerného platu", command=podla_platu).pack()

canvas.mainloop()
