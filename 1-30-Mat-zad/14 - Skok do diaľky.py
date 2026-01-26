subor = open("skok_do_dialky.txt","r",encoding="UTF-8")
riadky = subor.readlines()
krajiny = []
krajiny_vsetky = []
rovn_mena = []
naj_skok = 0
naj_meno = ""

for riadok in riadky:
    splitnute = riadok.split()
    krajiny_vsetky.append(splitnute[1])
    meno = splitnute[0]
    if splitnute[1] not in krajiny:
        krajiny.append(splitnute[1])
    
    for i in range (2, len(splitnute)):
        skok = int(splitnute[i])
        if skok > naj_skok:
            naj_skok = skok
            rovn_mena = [meno]
        elif skok == naj_skok:
              if meno not in rovn_mena:
                   rovn_mena.append(meno)

for k in krajiny:
        print("z krajiny: ",k, "je pocet: ", krajiny_vsetky.count(k))

print("sutaziaci su z:", ", ".join(krajiny))
print("Najlepší skok:", naj_skok, "urobili súťažiaci:", ", ".join(rovn_mena))