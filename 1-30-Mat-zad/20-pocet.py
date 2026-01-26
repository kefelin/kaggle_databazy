subor = open("tabulka_pocetnosti.txt","r",encoding="UTF-8")
pocty = [0]*26
for riadok in subor:
    print(riadok,end="")
    for znak in riadok:
        velke = znak.upper()
        if "A" <= velke <= "Z":
            pocty[ord(velke) -65] += 1
print()
print("Pocetnost jednotlivych pismen:")
for i in range(26):
    if pocty[i] != 0:
        print(chr(65+i),"-",pocty[i])
print("Nepouzite pismena:")
for i in range(26):
    if pocty[i] == 0:
        print(chr(65+i), end=" ")