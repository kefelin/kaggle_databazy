import random
vstup = open("virus.txt","r",encoding="UTF-8")
vst_riadky = vstup.readlines()
nahod3 = random.randint(1,2)
if nahod3 == 1:  #nahod porad viet
    random.shuffle(vst_riadky)
    print(nahod3)

pomiesane = []
for riadok in vst_riadky:
    nahod = random.randint(1,2)
    slova = riadok.split()
    print(nahod)
    nove_slova = []
    if nahod == 1:      #poradie slov v riadku
        random.shuffle(slova)
    for slovo in slova:
        if random.randint(1,3) == 1:
            slovo = slovo[:: -1]   #otocenie slov
        nove_slova.append(slovo)
    pomiesane.append(" ".join(nove_slova) + "\n")

with open("virus_vystup.txt", "w", encoding="UTF-8") as vystup:
    vystup.writelines(pomiesane)


