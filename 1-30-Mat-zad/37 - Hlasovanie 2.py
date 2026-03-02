subor = open("hlasovanie_1.txt", "r", encoding="utf-8")
riadky = subor.readlines()
subor.close()
pocet_sms = len(riadky)
print("Celkovy pocet sm:", pocet_sms)
sutaziaci = []

for i in range(5220, 5230):
    sutaziaci.append(str(i))
for cislo_sutaziaceho in sutaziaci:
    nazov_suboru = cislo_sutaziaceho + ".txt"
    vystup = open(nazov_suboru, "w", encoding="utf-8")
    poradie = 1
    for riadok in riadky:
        tel_cislo = riadok.strip()
        
        if tel_cislo == cislo_sutaziaceho:
            vystup.write(str(poradie) + "\n") 
        poradie = poradie + 1
    vystup.close()