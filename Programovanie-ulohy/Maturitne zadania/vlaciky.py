import tkinter
canvas = tkinter.Canvas(width=600, height=150, bg="black")
canvas.pack()

with open("vlacik.txt", "r") as subor1:
    riadok = subor1.read()
mesta = riadok.split()
pocet = 0  
text_id = None  

def animuj_text(x, y, text):
    global text_id
    if x < 600:
        canvas.move(text_id, 5, 0)  
        canvas.after(50, animuj_text, x + 5, y, text)  
    elif x >= 600:
        canvas.move(text_id, -600, 0)
        canvas.after(50, animuj_text)  

def spusti_animaciu(event):
    global pocet, text_id
    if pocet < len(mesta):
        text_id = canvas.create_text(-50, 75, text=mesta[pocet], font="Arial 35 bold", fill="red", anchor="w")
        animuj_text(-50, 75, mesta[pocet])
        pocet += 1
canvas.bind("<Button-1>", spusti_animaciu)
canvas.mainloop()
