subor = open("mena_zamestnancov.txt", "r", encoding="UTF-8")
subor2 = open("vystup.txt","w", encoding="UTF-8")
riadky = subor.readlines()
pocet_mien = 0
mena = []
priezv = []
mena_po, priezviska_od = 5, 5
naj_meno = ""

for riadok in riadky[0:mena_po]:
    pocet_mien += 1
    meno = riadok.strip()
    mena.append(meno)
    print(meno)

    if len(meno) > len(naj_meno):
        naj_meno = meno

for riadok in riadky[priezviska_od:]:
    priezviska = riadok.strip()
    priezv.append(priezviska)

max_dlzka = 0
for meno in mena:
    if len(meno) > max_dlzka:
        max_dlzka = len(meno)

for i in range(mena_po):
    meno = mena[i]
    priezvisko = priezv[i]

    rozdiel = max_dlzka - len(meno) + 4 
    medzery = ""
    for j in range(rozdiel):
        medzery = medzery + " "
    subor2.write(meno + medzery + priezvisko + "\n")

subor.close()
subor2.close()
print("Najdlhšie meno:", naj_meno)
print("Počet mien:", pocet_mien)
print("Mena:", ", ".join(mena))
