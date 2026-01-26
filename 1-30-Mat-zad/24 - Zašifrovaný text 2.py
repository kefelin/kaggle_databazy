import random
subor = open("vstupny_text.txt", "r", encoding="UTF-8")
subor2 = open("zasifrovany_text_2.txt", "w", encoding="UTF-8")
volba = input("Chces sifrovat alebo desifrovat? (s/d): ").strip().lower()
zoz = []
for riadok in subor:
    riadok = riadok.rstrip("\n")
    upraveny_riadok = ""
    if volba == "s":
        posun = random.randint(1, 25)
        upraveny_riadok += chr(posun + ord("a") - 1)
    else:
        if len(riadok) > 0:
            prvy_znak = riadok[0]
            posun = ord(prvy_znak) - ord("a") + 1
            riadok = riadok[1:]
        else:
            posun = 0
    for znak in riadok:
        if "a" <= znak <= "z":
            if volba == "s":
                nove = chr((ord(znak) - ord("a") + posun) % 26 + ord("a"))
            else:
                nove = chr((ord(znak) - ord("a") - posun) % 26 + ord("a"))
        else:
            nove = znak
        upraveny_riadok += nove
    zoz.append(upraveny_riadok)
vysledok = "\n".join(zoz)
print("Výsledok:\n")
print(vysledok)
subor2.write(vysledok)
subor.close()
subor2.close()