import random
vstup = input("zadaj tvojich 6 cisel: ")
vstup_split = vstup.split()
subor1 = open("Loteria_1.txt","r")
riadky = subor1.readlines()
moj_tip = []
for cis in vstup_split:
    moj_tip.append(int(cis))
cisla = []
for i in range (6):
    cislo = random.randint(1,49)
    while cislo in cisla:
        cislo = random.randint(1,49)
    cisla.append(cislo)
print("vybrate cisla: ",cisla)
uhadnute = []
for cis in moj_tip:
    if cis in cisla:
        uhadnute.append(cis)
print("Uhádnuté čísla:", "".join(str(uhadnute)))
print("Počet uhádnutých:", len(uhadnute))
jeden, dva ,tri ,stiry ,pat ,sest = 0,0,0,0,0,0
ucast = 0
for riadok in riadky:
    ucast += 1
    vyher = []
    cisla_riadok = riadok.split()
    cisla_int = []
    for i in cisla_riadok:
        cisla_int.append(int(i))
    cisla_riadok = cisla_int
    for cislo in cisla_riadok:
        if cislo in cisla:
            vyher.append(cislo)
    if len(vyher) == 1:
        jeden += 1
    elif len(vyher) == 2:
        dva += 1
    elif len(vyher) == 3:
        tri += 1
    elif len(vyher) == 4:
        stiry += 1
    elif len(vyher) == 5:
        pat += 1
    elif len(vyher) == 6:
        sest += 1
    print("Clovek:",ucast,".","ma vyher cisla:",vyher)
print("\nŠtatistika účastníkov:")
print("Presne 1 číslo uhádlo:", jeden)
print("Presne 2 čísla uhádlo:", dva)
print("Presne 3 čísla uhádlo:", tri)
print("Presne 4 čísla uhádlo:", stiry)
print("Presne 5 čísiel uhádlo:", pat)
print("Presne 6 čísiel uhádlo:", sest)