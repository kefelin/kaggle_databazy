vstup = open("hlasovanie_1.txt","r")
vstup2 = open("hlasovanie_vypadnuti.txt","r")
riadky = vstup.readlines()
riadky2 = vstup2.readlines()
dokopy = 0
vypadnuty = []
for riadok2 in riadky2:
    riadok2 = riadok2.strip()
    vypadnuty.append(riadok2)
print(vypadnuty)
nulty, prvy, druhy, treti, stvrti, piaty, siesty, siedmi, osmi, deviati = 0,0,0,0,0,0,0,0,0,0
for riadok in riadky:
    riadok = riadok.strip()
    if riadok in vypadnuty:
        continue
    dokopy += 1
    if riadok == "5220":
        nulty += 1
    elif riadok == "5221":
        prvy += 1
    elif riadok == "5222":
        druhy += 1
    elif riadok == "5223":
        treti += 1
    elif riadok == "5224":
        stvrti += 1
    elif riadok == "5225":
        piaty += 1
    elif riadok == "5226":
        siesty += 1
    elif riadok == "5227":
        siedmi += 1
    elif riadok == "5228":
        osmi += 1
    elif riadok == "5229":
        deviati += 1
vsetky_pocty = [nulty, prvy, druhy, treti, stvrti, piaty, siesty, siedmi, osmi, deviati]
vsetky_kody = ["5220", "5221", "5222", "5223", "5224", "5225", "5226", "5227", "5228", "5229"]
najm_pocet = -1
najm_kod = ""
for i in range(len(vsetky_pocty)):
    aktualny_kod = vsetky_kody[i]
    aktualny_pocet = vsetky_pocty[i]
    if aktualny_kod not in vypadnuty:
        if najm_pocet == -1 or aktualny_pocet < najm_pocet:
            najm_pocet = aktualny_pocet
            najm_kod = aktualny_kod
print(nulty, prvy, druhy, treti, stvrti, piaty, siesty, siedmi, osmi, deviati)
print("najmenej mal: ", najm_kod,"s poctom hlasov: ", najm_pocet)
print("Celkovy pocet SMS je:",dokopy)