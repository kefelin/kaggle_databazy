veta = input("Zadaj vetu: ")

sifrovanie = [(" "), ("A","B","C"),("D","E","F"),("G","H","I"),("J","K","L"),("M","N","O"),("P","Q","R"),("S","T","U"),("V","W","X"),("Y","X")]
vystup = []

for znak in veta.upper():
   cislo = 0
   naslo = False
   for skupina in sifrovanie:
      pozicia = 1
      for text in skupina:
         if text == znak and not naslo:
            vystup.append(str(cislo) * pozicia)
            naslo = True
         pozicia += 1
      cislo += 1

print(" ".join(vystup))