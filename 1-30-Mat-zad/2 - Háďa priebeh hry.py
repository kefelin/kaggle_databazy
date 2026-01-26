def nacitaj_hry(subor):
    hry = []
    with open(subor, "r") as f:
        for riadok in f:
            riadok = riadok.strip()
            if riadok != "":
                hry.append(riadok)
    return hry

def komprimuj_riadok(riadok):
    if riadok == "":
        return ""
    vysledok = ""
    aktualny = riadok[0]
    pocet = 1
    for znak in riadok[1:]:
        if znak == aktualny:
            pocet = pocet + 1
        else:
            vysledok = vysledok + aktualny + str(pocet)
            aktualny = znak
            pocet = 1
    vysledok = vysledok + aktualny + str(pocet)
    return vysledok

hry = nacitaj_hry("hada.txt")

pocet_hier = len(hry)

najdlhsia = 0
for hra in hry:
    if len(hra) > najdlhsia:
        najdlhsia = len(hra)

with open("had_skrat.txt", "w") as f:
    for hra in hry:
        f.write(komprimuj_riadok(hra) + "\n")

print("Pocet hier:", pocet_hier)
print("Najdlhsia hra mala krokov:", najdlhsia)