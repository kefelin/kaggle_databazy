import tkinter
import random

canvas = tkinter.Canvas(width=600, height=200, bg="white")
canvas.pack()

oznacenia = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

puknuty = random.choice(oznacenia)

stredy = []     # strdy taniery
polomer = 25
for i in range(10):
    x = 50 + i * 55
    y = 100
    stredy.append((x, y))

kliknutia = {}  # pocita kliknuti tanietov individ
for pismeno in oznacenia:
    kliknutia[pismeno] = 0

def klik(event):
    x = event.x
    y = event.y

    for i in range(10):
        stred_x, stred_y = stredy[i]
        dx = x - stred_x
        dy = y - stred_y
        if dx * dx + dy * dy <= polomer * polomer:
            meno = oznacenia[i]
            kliknutia[meno] = kliknutia[meno] + 1
            if meno == puknuty:
                canvas.delete("all")
                canvas.create_text(300, 80, text="Gratulujem, našiel si puknutý tanier!", font="Arial 16 bold", fill="green")
                text2 = "Viackrát si klikol na taniere:"
                nieco_nasiel = False
                for pismeno in oznacenia:
                    if kliknutia[pismeno] >= 2:
                        text2 = text2 + " " + pismeno + ","
                        nieco_nasiel = True

                if nieco_nasiel:
                    if text2.endswith(","):
                        text2 = text2[:-1]
                else:
                    text2 = "Viackrát si neklikol na žiadny tanier."
                canvas.create_text(300, 120, text=text2, font="Arial 14", fill="red")
                canvas.unbind("<Button-1>")
            return

#vykreslenie tan
for i in range(10):
    x, y = stredy[i]
    canvas.create_oval(x - polomer, y - polomer, x + polomer, y + polomer,fill="lightblue", outline="black")
    canvas.create_text(x, y, text=oznacenia[i], font="Arial 14 bold")

canvas.bind("<Button-1>", klik)
canvas.mainloop()
