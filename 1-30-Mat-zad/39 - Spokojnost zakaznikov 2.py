import tkinter
subor = open("spokojnost_1.txt", "r", encoding="utf-8")
riadky = subor.readlines()
subor.close()
pocet_nespokojnych_celkovo = 0
nespokojni_podla_hodin = [0] * 24
for riadok in riadky:
    casti = riadok.strip().split()
    cas = casti[0]
    odpoved = casti[1] 
    cas_split = cas.split(":")
    hodina = int(cas_split[0])
    if odpoved == "nie":
        pocet_nespokojnych_celkovo = pocet_nespokojnych_celkovo + 1
        nespokojni_podla_hodin[hodina] = nespokojni_podla_hodin[hodina] + 1

print("elkov pocet negativ vyjadreni: ", pocet_nespokojnych_celkovo)
max_nespokojnych = 0
hodina_max = 0
for h in range(24):
    if nespokojni_podla_hodin[h] > max_nespokojnych:
        max_nespokojnych = nespokojni_podla_hodin[h]
        hodina_max = h

print("najviac nespokojn zakaznikov bolo v hodine: ", hodina_max)
print("ich absolutny pocet bol:", max_nespokojnych)
print("pocty nespokojny v jendotli hodinach:")
for h in range(24):
    if nespokojni_podla_hodin[h] > 0:
        print("Hodina", h, ":", nespokojni_podla_hodin[h])
canvas = tkinter.Canvas(width=480, height=520, bg="white")
canvas.pack()

x = 10
sirka_stlpca = 18
for h in range(24):
    vyska_stlpca = nespokojni_podla_hodin[h] * 20
    if nespokojni_podla_hodin[h] > 0:
        canvas.create_rectangle(x, 500, x + sirka_stlpca, 500 - vyska_stlpca, fill="red")
    text_hodina = str(h)
    if h < 10:
        text_hodina = "0" + text_hodina   
    canvas.create_text(x + sirka_stlpca//2, 510, text=text_hodina, font="Arial 8")
    x = x + 20
canvas.mainloop()