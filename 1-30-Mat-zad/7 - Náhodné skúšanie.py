import random
pocet_studentov = int(input("Zadaj počet študentov: "))
pocet_otazok = int(input("Zadaj počet otázok: "))

if pocet_otazok < pocet_studentov:
    print("Chyba: otázok je menej ako študentov")
else:
    studenti = [] #tu zoznam sa tvori pre tych 
    cislo = 1
    while cislo <= pocet_studentov:
        studenti.append(cislo)
        cislo = cislo + 1

    otazky = []   # tu zas zoznam pre otazky    
    cislo = 1
    while cislo <= pocet_otazok:
        otazky.append(cislo)
        cislo = cislo + 1

    poradie_studentov = []          # random poradie studentov 
    while len(poradie_studentov) < pocet_studentov:
        nahodne = random.randint(1, pocet_studentov)
        if nahodne not in poradie_studentov:
            poradie_studentov.append(nahodne)

    vybrate_otazky = []  #nahod otaz ale nemaly by sa uz opakovat dufam...
    while len(vybrate_otazky) < pocet_studentov:
        nahodne = random.randint(1, pocet_otazok)
        if nahodne not in vybrate_otazky:
            vybrate_otazky.append(nahodne)

    print("Náhodné poradie študentov a ich otázky:")
    i = 0
    while i < pocet_studentov:
        print(str(i+1) + ". študent", poradie_studentov[i], ", otázka", vybrate_otazky[i])
        i = i + 1
