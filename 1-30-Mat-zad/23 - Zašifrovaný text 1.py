subor = open("vstupny_text.txt","r",encoding="UTF-8")
subor2 = open("zasifrovany_text_1.txt","w",encoding="UTF-8")
volba = input("Chces sifrovat alebo desifrovat? (s/d): ").strip().lower()
kluc = input("Zadaj kluc: ").strip()
riadky = subor.read()
zasifrovane = ""
zoz = []
for i in range(len(riadky)):
    znak = riadky[i]
    if "a" <= znak <= "z":
        posun = ord(kluc[i % len(kluc)]) - ord("a") + 1
        if volba == "s":
            nove = chr((ord(znak) - ord("a") + posun) % 26 + ord("a"))
        else:
            nove = chr((ord(znak) - ord("a") - posun) % 26 + ord("a"))
    else:
        nove = znak
    zasifrovane = zasifrovane + nove
zoz.append(zasifrovane)
print("Výsledok:\n")
print(zasifrovane)
print("Zoznam s výsledkom:", zoz)
subor2.write("Výsledný text: \n" + "".join(zoz) + "\n\nPoužitý kluc: \n" + kluc + "\n")