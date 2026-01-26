vstup = open("poprehadzovany_text1_vstup.txt","r",encoding= "UTF-8")
vystup = open("poprehadzovany_text1_vystup.txt","w",encoding= "UTF-8")
import random
slova = []
print("Ukážka vstupného textového súboru: " + "\n")

riadky = vstup.readlines()

for riadok in riadky:
    print(riadok, end="")

print("\n\nUkážka výstupného textového súboru: \n")
for riadok in riadky:
    split_raidok = riadok.split()
    poprehadzovane_slova = []
    for slovo in split_raidok:
        prve = slovo[0]
        posl = slovo[-1]
        stred = list(slovo[1:-1])
        random.shuffle(stred)
        stred_pom = "".join(stred)
        nove_slovo = prve + stred_pom + posl
        poprehadzovane_slova.append(nove_slovo)
    upraveny_raidok = " ".join(poprehadzovane_slova)
    print(upraveny_raidok)
    vystup.write(upraveny_raidok + "\n")

