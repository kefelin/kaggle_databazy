subor = open ("sutaz_v_behu.txt","r")
sportovci = []
for riadok in subor:
    split = riadok.strip().split()
    sportovci.append((split[0], int(split[1])))
print("pocet zucast je: ", len(sportovci))
naj = sportovci [0] [1]
for prvok in sportovci:
    if prvok[1] < naj:
        naj = prvok[1]
        vitaz = prvok[0]
print("naj sport: ", vitaz, "s casom: ", naj // 60, "min", naj % 60, "sek")