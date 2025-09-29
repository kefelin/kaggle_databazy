import tkinter
import random
canvas = tkinter.Canvas(height=300,width=600,bg="black")
canvas.pack()
frekvenice = (32,64,125,250,500,"1K","2K","4K","8K","16K")
x=20
for i in frekvenice:
    x += 50
    canvas.create_text(x,275,text=i,fill="green",font="Arial 15 bold")

def vykresli_stlpce ():
    canvas.delete("stlpce")
    x1 = 70
    for i in range(10):
        y = random.randint(50,100)
        canvas.create_rectangle(x1,250,x1+20,200-y,fill = "green",tags="stlpce")
        x1 += 50
    canvas.after(250, vykresli_stlpce)

vykresli_stlpce()
canvas.mainloop()
