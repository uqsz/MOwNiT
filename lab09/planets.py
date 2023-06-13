import pygame
import numpy as np
import tkinter as tk


def r(x, y):
    return np.sqrt(x**2+y**2)


def fx(x, y):
    return -x*(x**2+y**2)**(-3/2)


def fy(x, y):
    return -y*(x**2+y**2)**(-3/2)


def euler_poljawny(t_end, h, e):
    Tx = [1-e]
    Ty = [0]
    Tvx = [0]
    Tvy = [np.sqrt((1+e)/(1-e))]
    t = 0
    i = 0
    while t < t_end:
        Tvx.append(Tvx[i]+h*fx(Tx[i], Ty[i]))
        Tx.append(Tx[i]+h*Tvx[i+1])
        Tvy.append(Tvy[i]+h*fy(Tx[i], Ty[i]))
        Ty.append(Ty[i]+h*Tvy[i+1])
        i += 1
        t += h
    return Tx, Ty

def run(e,h,v):
    x, y = euler_poljawny(40, 10**(-h), e)

    x = np.array(x)*400+1000
    y = np.array(y)*400+500

    x=x[::v]
    y=y[::v]

    z=list(zip(x,y))

    pygame.init()
    screen = pygame.display.set_mode((1600, 1000))
    pygame.display.set_caption("Animacja ruchu obiektu")
    black = (0, 0, 0)
    blue = (0, 0, 255)
    gold = (255, 215, 0)
    red = (255, 0, 0)
    obj_radius = 15
    clock = pygame.time.Clock()

    running = True
    idx = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill(black)
        pygame.draw.circle(screen, gold, (1000, 500), 30)
        pygame.draw.lines(screen, red, False, z[:idx+2], 1)
        pygame.draw.circle(screen, blue, (x[idx], y[idx]), obj_radius)
        pygame.display.flip()
        clock.tick(200)
        idx += 1
        if idx >= len(x)-2:
            idx = 0

    pygame.quit() 

# run(0.95,3,9)

# Funkcja do obsługi przycisku "OK"
def ok_button_click():
    e = float(entry1.get())
    h = float(entry2.get())
    v = int(entry3.get())
    run(e, h, v)

    

# Utworzenie okna
window = tk.Tk()

# Tworzenie etykiet i pól wprowadzania dla parametrów
label1 = tk.Label(window, text="Zakrzywienie orbity (0,1):")
label2 = tk.Label(window, text="Dokładność metody (1,4):")
label3 = tk.Label(window, text="Szybkość [1,2,3,4...]:")

entry1 = tk.Entry(window)
entry2 = tk.Entry(window)
entry3 = tk.Entry(window)

# Ustawienie elementów w siatce (grid)
label1.grid(row=0, column=0, sticky="e")
label2.grid(row=1, column=0, sticky="e")
label3.grid(row=2, column=0, sticky="e")

entry1.grid(row=0, column=1)
entry2.grid(row=1, column=1)
entry3.grid(row=2, column=1)

# Tworzenie przycisku "OK" do zatwierdzenia wprowadzonych wartości
ok_button = tk.Button(window, text="Uruchom", command=ok_button_click)
ok_button.grid(row=3, columnspan=2)

# Uruchomienie pętli głównej programu
window.mainloop()