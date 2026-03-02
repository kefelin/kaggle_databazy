subor = open("spokojnost_1.txt", "r", encoding="UTF-8")
riadky = subor.readlines()
subor.close()
pocet_vyjadreni = len(riadky)
print("Celkovy pocet vyjadreni:", pocet_vyjadreni)
spokojni_podla_hodin = [0] * 24
nespokojni_podla_hodin = [0] * 24
vsetci_podla_hodin = [0] * 24

for riadok in riadky:
    casti = riadok.strip().split()
    cas = casti[0]
    odpoved = casti[1]
    cas_split = cas.split(":")
    hodina = int(cas_split[0])
    vsetci_podla_hodin[hodina] = vsetci_podla_hodin[hodina] + 1
    if odpoved == "ano":
        spokojni_podla_hodin[hodina] = spokojni_podla_hodin[hodina] + 1
    else:
        nespokojni_podla_hodin[hodina] = nespokojni_podla_hodin[hodina] + 1

max_spokojnych = 0
hodina_max_spokojnych = 0
max_nespokojnych = 0
hodina_max_nespokojnych = 0
for h in range(24):
    if spokojni_podla_hodin[h] > max_spokojnych:
        max_spokojnych = spokojni_podla_hodin[h]
        hodina_max_spokojnych = h 
    if nespokojni_podla_hodin[h] > max_nespokojnych:
        max_nespokojnych = nespokojni_podla_hodin[h]
        hodina_max_nespokojnych = h

print("najviac spoko bolo v hodine: ", hodina_max_spokojnych, "Pocet:", max_spokojnych)
print("najviac nespok bolo v hodine: ", hodina_max_nespokojnych, "Pocet:", max_nespokojnych)
print("percenta spokojnosti:")
for h in range(24):
    if vsetci_podla_hodin[h] > 0:
        percento = (spokojni_podla_hodin[h] / vsetci_podla_hodin[h]) * 100
        print("Hodina", h, ":", round(percento, 2), "%")