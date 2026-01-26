import random
subor = open("poprehadzovany_text2.txt","r",encoding="UTF-8")
subor1 = open("poprehadzovany_text_vystup2.txt","w",encoding="UTF-8")
riadky = subor.readlines()
pozor_znaky = [".", ",", "?", "!", ":", ";", "…", "-", "\"", "„", "“", "'", "(", ")", "[", "]", "{", "}", "/", "\\", "*", "#", "%", "@"]
prefix = ""
suffix = ""
nove = []
def pomiesaj ():
    for riadok in riadky:
        split_riadok = riadok.strip().split()
        poprehadzovane_slova = [] 
        for slovo in split_riadok:
            suffix = ""
            prefix = ""
            zaciatok = 0
            koniec = len(slovo)
            if slovo[0] in pozor_znaky:
                zaciatok = 1
                prefix = slovo[0]
            if slovo[-1] in pozor_znaky:
                koniec = koniec - 1
                suffix = slovo[-1]
            jadro = slovo[zaciatok:koniec]
            if len(jadro) > 2:
                stred = list(jadro[1:-1])
                random.shuffle(stred)
                stred_pom = "".join(stred)
                nove_slovo = prefix + jadro[0] + stred_pom + jadro[-1] + suffix
            else:
                nove_slovo = prefix + jadro + suffix
            poprehadzovane_slova.append(nove_slovo)
        upraveny_riadok = " ".join(poprehadzovane_slova)
        subor1.write(upraveny_riadok + "\n")
        nove.append(upraveny_riadok)
pomiesaj()
print("\nPoprehadzovany text: \n")
for riad in nove:
    print(riad)
subor.close()
subor1.close





