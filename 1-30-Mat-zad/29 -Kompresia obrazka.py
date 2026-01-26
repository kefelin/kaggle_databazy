vstup = open("kompresia_obrazka_1.txt","r",encoding= "UTF-8")
vystup = open("kompresia_obrazka_vystup.txt","w",encoding= "UTF-8")
riadok1 = vstup.readline()
riadok_up = riadok1.strip().split()
sirka = int(riadok_up[0])
vyska = int(riadok_up[1])
vystup.write(str(sirka)+" "+ str(vyska) + "\n")
def spracuj_riadok (sirka , vyska):
    for riadok in vstup:
        vysledne = []
        jedna_poc = 0
        nula_poc = 0
        if riadok and riadok[0] == "1":
            vysledne.append(0)
        for znak in riadok:
            if znak == "0":
                if jedna_poc > 0:
                    vysledne.append(jedna_poc)
                jedna_poc = 0
                nula_poc += 1
            elif znak == "1":
                if nula_poc > 0:
                    vysledne.append(nula_poc)
                nula_poc = 0
                jedna_poc += 1
        if jedna_poc > 0:
            vysledne.append(jedna_poc)
        elif nula_poc > 0:
            vysledne.append(nula_poc)
        print(vysledne)
        riaok_zapis = ""
        for cislo in vysledne:
            riaok_zapis += str(cislo) + " "
        vystup.write(riaok_zapis + "\n")
spracuj_riadok(sirka,vyska)
vystup.close()
vstup.close()