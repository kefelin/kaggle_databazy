class Vlajka:
    def __init__(self, x, y, sirka, vyska, zvislo, farby):
        self.zvislo = zvislo
        self.farby = farby
        self.x = x
        self.y = y
        self.sirka = sirka
        self.vyska = vyska
        self.obdlzniky = []

    def kresli(self, cnv):
        for i in range(len(self.obdlzniky)):
            cnv.delete(self.obdlzniky[i])
        n = len(self.farby)

        for i in range(n):
            x = self.x - self.sirka / 2
            y = self.y - self.vyska / 2

            if self.zvislo:
                obdlznik = cnv.create_rectangle(x+i*self.sirka/n, y,x+(i+1)*self.sirka/n,y+self.vyska, width=0,fill=self.farby[i])
                self.obdlzniky.append(obdlznik)
            else :
                obdlznik = cnv.create_rectangle(x, y+i*self.vyska/n,x+self.sirka,y+(i+1)*self.vyska/n,width=0, fill=self.farby[i])
                self.obdlzniky.append(obdlznik)

    def zoom(self, pomer):
        self.sirka *= pomer
        self.vyska *= pomer

    def generuj_nahodnu():
        import random
        x = random.randint(100, 700)
        y = random.randint(100, 500)
        sirka = random.randint(50, 300)
        vyska = random.randint(50, 300)
        zvislo = random.choice([True, False])
        farby = []
        for i in range(random.randint(2, 5)):
            farba = '#%06x' % random.randint(0, 0xFFFFFF)
            if farba not in farby:
                farby.append(farba)
        return Vlajka(x, y, sirka, vyska, zvislo, farby)