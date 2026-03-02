subor = open("spokojnost_1.txt", "r", encoding="utf-8")
riadky = subor.readlines()
subor.close()
pocet_vyjadreni = len(riadky)
vyjadrenia_v_hodinach = [0] * 24
vyjadrenia_v_dnoch = []
predchadzajuci_cas_v_minutach = 9999
pocet_reakcii_aktualny_den = 0

for riadok in riadky:
    casti = riadok.strip().split()
    cas_text = casti[0]
    cas_split = cas_text.split(":")
    hodina = int(cas_split[0])
    minuta = int(cas_split[1])
    aktualny_cas_v_minutach = hodina * 60 + minuta
    vyjadrenia_v_hodinach[hodina] = vyjadrenia_v_hodinach[hodina] + 1
    if aktualny_cas_v_minutach < predchadzajuci_cas_v_minutach:
        if pocet_reakcii_aktualny_den > 0:
            vyjadrenia_v_dnoch.append(pocet_reakcii_aktualny_den)
        pocet_reakcii_aktualny_den = 1
    else:
        pocet_reakcii_aktualny_den = pocet_reakcii_aktualny_den + 1
    predchadzajuci_cas_v_minutach = aktualny_cas_v_minutach
if pocet_reakcii_aktualny_den > 0:
    vyjadrenia_v_dnoch.append(pocet_reakcii_aktualny_den)
for i in range(len(vyjadrenia_v_dnoch)):
    print(str(i + 1) + ". den - pocet reakcii:" + str(vyjadrenia_v_dnoch[i]))
print("pocet vsetkych vyjadreni: ", pocet_vyjadreni)

for h in range(24):
    if vyjadrenia_v_hodinach[h] > 0:
        print("Hodina:" + str(h) + " reakcii zakaznikiov :" + str(vyjadrenia_v_hodinach[h]))
print("pocet dni: ", len(vyjadrenia_v_dnoch))