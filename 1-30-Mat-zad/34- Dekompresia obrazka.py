import tkinter
subor1 = open("dekompresia_obrazka_1.txt", "r", encoding="utf-8")
subor2 = open("dekompresia_obrazka_vystup.txt", "w", encoding="utf-8")
prvy_riadok = subor1.readline()
prvy_riadok_split = prvy_riadok.strip().split()

sirka = int(prvy_riadok_split[0])
vyska = int(prvy_riadok_split[1])
print("Sirka:", sirka)
print("Vyska:", vyska)
print("Pocet vsetkych bodov:", sirka * vyska)

subor2.write(str(sirka) + " " + str(vyska) + "\n")
def spracuj_riadok(riadok_cisiel):
    vystupny_retazec = ""
    casti = riadok_cisiel.strip().split()
    farba = "0" 
    for cislo in casti:
        pocet = int(cislo)
        vystupny_retazec = vystupny_retazec + (farba * pocet)
        if farba == "0":
            farba = "1"
        else:
            farba = "0"     
    return vystupny_retazec

riadky = subor1.readlines()
for i in riadky:
    novy_riadok = spracuj_riadok(i)
    subor2.write(novy_riadok + "\n")
subor1.close()
subor2.close()
print("dokncene")