import tkinter as tk

def otvor_canvas2():
    #vytvorenie noveho okna
    okno2 = tk.Toplevel (root)
    okno2.title("canvas 2")

    # vytvori novy canvas  tomto okne
    canvas2 = tk.Canvas(okno2, width = 400, height = 300, bg="lightyellow", scrollregion=(0,-200,0,1200))
    canvas2.pack(padx=10, pady=10)

    frame = tk.Frame(okno2)
    frame.pack(fill="both",expand=True)

    label1 = tk.Label(frame, text="Tu máme ukažku skrolovania")
    label1.pack()


    v_scroll = tk.Scrollbar(frame, orient="vertical",command=canvas2.yview)
    v_scroll.pack(side="right",fill="y")
    canvas2.configure(yscrollcommand=v_scroll.set)

    inner_frame = tk.Frame(canvas2, bg="lightyellow")
    canvas2.create_window((0, 0), window=inner_frame, anchor="nw")

    canvas2.create_oval(50,50,200,200,fill="orange",outline="black")
    canvas2.create_text(200,250,text="Toto je canvas2",font="Arial 15 bold")

def otvor_canvas3():
    okno3 = tk.Toplevel (root)
    okno3.title ("Canvas 3")

    canvas3 = tk.Canvas(okno3, width=400, height = 300, bg = "green")
    canvas3.pack(padx=10, pady=10)

    frame = tk.Frame(okno3)
    frame.pack(fill="both",expand=True)

    label2 = tk.Label(frame, text="Tu máme ukažku Radio Buttonov")
    label2.pack()

    vyber0 = tk.StringVar(value="")
    vyber1 = tk.StringVar(value="")

    def ano ():
        print("Áno")
    def nie():
        print("Drbe ti?")

    radiobut1 = tk.Radiobutton(canvas3, text="Áno", variable=vyber0, value="A",command=ano)
    radiobut2 = tk.Radiobutton(canvas3, text="Nie", variable=vyber1, value="B",command=nie)

    canvas3.create_window(50,250, window=radiobut1)
    canvas3.create_window(350,250, window=radiobut2)

    canvas3.create_text(200,150,text="Zaznač: Áno", font="Arial 15 bold", fill= "White")

def otvor_canvas4():
    okno4 = tk.Toplevel (root)
    okno4.title ("Canvas 4")

    canvas4 = tk.Canvas(okno4, width=400, height = 300, bg = "blue")
    canvas4.pack(padx=10, pady=10)

    frame = tk.Frame(okno4)
    frame.pack(fill="both",expand=True)

    label3 = tk.Label(frame, text="Tu máme ukažku Check Boxov")
    label3.pack()

    vyber2 = tk.StringVar(value="")
    vyber3 = tk.StringVar(value="")

    def chleba ():
        print("Chlebicok")
    def rozok ():
        print("rozok")

    checkbut1 = tk.Checkbutton(canvas4, text="Chleba", variable=vyber2,command=chleba)
    checkbut2 = tk.Checkbutton(canvas4, text="Rožok", variable=vyber3,command=rozok)

    canvas4.create_window(50,250, window=checkbut1)
    canvas4.create_window(350,250, window=checkbut2)

    canvas4.create_text(200,150,text="Čo máš radšej?", font="Arial 15 bold", fill= "White")

def otvor_canvas5():
    okno5 = tk.Toplevel (root)
    okno5.title ("Canvas 5")

    canvas5 = tk.Canvas(okno5, width=400, height = 300, bg = "pink")
    canvas5.pack(padx=10, pady=10)

    frame = tk.Frame(okno5)
    frame.pack(fill="both",expand=True)

    label3 = tk.Label(frame, text="Tu máme ukažku Scale")
    label3.pack()

    hodnota1 = tk.IntVar(value="15")

    def aktualizuj_font(value):
        canvas5.itemconfig(text_uloz, font=("Arial", int(value), "bold")) #itemconfig komand aby si vedel zmenit vlastsnost dakej veci vytvorenej

    scale = tk.Scale(canvas5, from_ = 1, to=250, orient="horizontal", length=400, variable=hodnota1,command=aktualizuj_font)
    canvas5.create_window(200,285, window=scale)

    text_uloz = canvas5.create_text(200, 125, text="Zveč ma", font=("Arial", hodnota1.get(), "bold"))


def otvor_canvas6():
    okno6 = tk.Toplevel (root)
    okno6.title ("Canvas 6")

    canvas6 = tk.Canvas(okno6, width=400, height = 300, bg = "green")
    canvas6.pack(padx=10, pady=10)

    frame = tk.Frame(okno6)
    frame.pack(fill="both",expand=True)

    label4 = tk.Label(frame, text="Tu máme ukažku Listbox")
    label4.pack()

    listbox = tk.Listbox(canvas6, height = 5, width= 20, selectmode="single")
    for item in ["Jablko", "Banán", "Hruška", "Pomaranč", "Kiwi"]:
        listbox.insert(tk.END, item)

    canvas6.create_window(200, 150, window=listbox)

def otvor_canvas7():
    okno7 = tk.Toplevel (root)
    okno7.title ("Canvas 7")

    canvas7 = tk.Canvas(okno7, width=400, height = 300, bg = "Orange")
    canvas7.pack(padx=10, pady=10)

    frame = tk.Frame(okno7)
    frame.pack(fill="both",expand=True)

    label4 = tk.Label(frame, text="Tu máme ukažku SpinBoxu")
    label4.pack()

    def tocky ():
        print(spin.get())

    spin =  tk.Spinbox(canvas7, from_=1, to=10, width=5, command=tocky)
    canvas7.create_window(200,250, window=spin)


#hlavne okno
root = tk.Tk()
root.title("Hlavne okno")

#vytvorime menu
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

#pridame menu polozku "zobrazit"
zobrazit_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label = "Zobrazit", menu = zobrazit_menu)

zobrazit_menu.add_command(label="Canvas2",command = otvor_canvas2)
zobrazit_menu.add_command(label="Canvas3",command = otvor_canvas3)
zobrazit_menu.add_command(label="Canvas4",command = otvor_canvas4)
zobrazit_menu.add_command(label="Canvas5",command = otvor_canvas5)
zobrazit_menu.add_command(label="Canvas6",command = otvor_canvas6)
zobrazit_menu.add_command(label="Canvas7",command = otvor_canvas7)

#hlavny canvas v hlavnom okne
canvas1 = tk.Canvas(root, width=400,height=300, bg="white")
canvas1.pack(padx=10, pady=10)
canvas1.create_rectangle (150,50,250,200,fill = "lightblue", outline = "blue")
canvas1.create_text(200,250,text="Toto je canvas1",font="Arial 15 bold")

canvas1.mainloop()