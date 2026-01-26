subor = open("tabulka_pocetnosti.txt","r",encoding="UTF-8")
znaky = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
pouzite_znaky = []
nepouzite = []
print("Text z txt suboru:"+"\n")
for riadok in subor:
    print(riadok,end="")
    for znak in riadok:
        if znak in znaky:
            pouzite_znaky.append(znak)

for z in pouzite_znaky:
    print(z,":", str(pouzite_znaky.count(z)))

for z in znaky:
    if z not in pouzite_znaky:
        nepouzite.append(z)
print("nepouzite znaky: ",", ".join(nepouzite))