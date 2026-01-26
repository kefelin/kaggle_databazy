subor = open("bus_vytazenost.txt","r")
kapacita = int(subor.readline())
zoznam = []
pretazene = []
pocet = 0
naj = 0
for riadok in subor:
    udaje = riadok.strip().split()
    if len(udaje) == 3:
        nazov = udaje[2]
    else:
        nazov = udaje[2] + " " + udaje[3]
    pocet += int(udaje[0])
    pocet -= int(udaje[1])
    if pocet > kapacita:
        pretazene.append(nazov)
        if pocet - kapacita > naj:
            naj = pocet - kapacita
        zoznam.append(nazov)
print ("kapacita je:", kapacita)
print("pocet zastvok je: ", len(zoznam))
print("zastavky na trase: ", end= "")

for zastavka in zoznam:
    print(zastavka, end=", ")
print()
print("autobus bol preplnenu po pov kapacitu po vyjedeni zomzastavok: ")
for zastavka in pretazene:
    print(zastavka)
print("najvacs pretazenie o: ", naj,"ludi")