text = open("objednane_jedla.txt", "r", encoding="utf-8")

zaznamy = []
for riadok in text:
    riadok = riadok.strip()
    if riadok != "":
        casti = riadok.split()
        cislo = casti[0]
        farba = casti[1]
        zaznamy.append([cislo, farba])

text.close()

print("Počet objednaných jedál:", len(zaznamy))

zelene = 0
cervene = 0
modre = 0
oranzova = 0

for z in zaznamy:
    if z[1] == "z":
        zelene += 1
    if z[1] == "č":
        cervene += 1
    if z[1] == "m":
        modre += 1
    if z[1] == "o":
        oranzova += 1

print("Zelené jedlá:", zelene)
print("Červené jedlá:", cervene)
print("Modré jedlá:", modre)
print("Oranžové jedlá:", oranzova)

najviac = "z"
pocet = zelene

if cervene > pocet:
    najviac = "č"
    pocet = cervene
if modre > pocet:
    najviac = "m"
    pocet = modre

if pocet >= 20:
    print("Najviac objednané jedlo:", najviac, "počet:", pocet)
else:
    print("Žiadne jedlo si neobjednalo aspoň 20 stravníkov")
