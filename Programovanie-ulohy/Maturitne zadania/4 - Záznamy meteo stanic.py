text = open("meteo_stanice.txt", "r", encoding="utf-8")

zaznamy = []
for riadok in text:
    riadok = riadok.strip()
    if riadok != "":
        casti = riadok.split()
        kod = casti[0]
        datum = casti[1]
        cas = casti[2]
        teplota_str = casti[3].replace(",", ".")
        teplota_str = teplota_str.replace("−", "-")
        teplota = float(teplota_str)
        pocasie = casti[4]
        zaznamy.append([kod, datum, cas, teplota, pocasie])
text.close()

print("Počet meraní:", len(zaznamy))

teploty = []
for i in zaznamy:
    teploty.append(i[3])

print("Teploty:", teploty)

najvyssia = teploty[0]
najnizsia = teploty[0]

for k in teploty:
    if k > najvyssia:
        najvyssia = k
    if k < najnizsia:
        najnizsia = k

print("Najvyššia teplota:", najvyssia)
print("Najnižšia teplota:", najnizsia)

for j in zaznamy:
    if j[3] == najvyssia:
        print("Stanica s najvyššou teplotou:", j[0])
    if j[3] == najnizsia:
        print("Stanica s najnižšou teplotou:", j[0])
