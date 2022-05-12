import math
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import scipy.integrate as integrate
import numpy as np
import scipy.stats

V = 77
n = 10
A = 5.20
B = 11.50


def binom():
    f = open("binom.txt", "r")
    binom = [0]*200
    i = 0
    for line in f.readlines():
        binom[i] = int(line.replace('\n', ''))
        i += 1
    print(binom)
    binom.sort()
    print(binom)

    xi = []
    ni = []
    wi = []
    for j in range(200):
        if not j == 199:
            if not binom[j] == binom[j+1]:
                xi.append(binom[j])
                ni.append(binom.count(binom[j]))
        else:
            xi.append(binom[j])
            ni.append(binom.count(binom[j]))

    for j in range(len(ni)):
        wi.append(round(ni[j]/200, 5))

    l = len(xi) - 2
    #--------------------------

    u1 = 0
    p = 0
    for j in range(len(xi)):
        u1 += round(xi[j] * wi[j], 5)
    p = round(u1/n, 5)

    #--------------------------

    xiBuf = xi[0]

    for j in range(xi[0]):
        xi.insert(0, xiBuf - j - 1)
        ni.insert(0, 0)
        wi.insert(0, 0)

    pi = []
    for i in range(len(xi)):
        pi.append(math.comb(len(xi) - 1, i) * pow(p, i) * pow((1 - p), len(xi) - i - 1))

    for i in range(len(pi)):
        pi[i] = round(pi[i], 5)

    wipi = []
    for j in range(len(wi)):
        wipi.append(round(abs(wi[j] - pi[j]), 5))

    roundNpi = []
    for j in range(len(wipi)):
        roundNpi.append(round(200*pow(wipi[j], 2)/pi[j], 5))

    vib_sr = 0
    for i in range(len(xi)):
        vib_sr += xi[i] * wi[i]

    vib_dis = 0
    for i in range(len(xi)):
        vib_dis += pow(xi[i], 2)*wi[i]

    print(xi)
    print(ni)
    print(wi)
    print(pi)
    print(sum(pi))
    print(wipi)
    print(max(wipi))
    print(roundNpi)
    print(sum(roundNpi))
    #print("Выборочное среднее: ", round(vib_sr, 5))
    #print("Выборочная дисперсия: ", round(vib_dis, 5))
    #print("Выборочное значение криерия XB^2", sum(roundNpi))

    Xkr = 0
    if l == 4:
        Xkr = 9.5
    elif l == 5:
        Xkr = 11.1
    elif l == 6:
        Xkr = 12.6
    elif l == 7:
        Xkr = 14.1
    elif l == 8:
        Xkr = 15.5
    print("Выборочное значение критерия Xkr^2", Xkr)
    if Xkr >= sum(roundNpi):
        print("Не противоречит")
    else:
        print("Противоречит")

    plt.axis([0, xi[len(xi)-1], 0, 1])
    plt.plot(xi, wi)
    plt.plot(xi, pi, color="red")

    subplot = plt.subplot()
    subplot.tick_params(which='both', width=1)
    subplot.tick_params(which='major', length=7)

    subplot.minorticks_on()
    subplot.xaxis.set_major_locator(tick.MultipleLocator(1))
    subplot.xaxis.set_minor_locator(tick.MultipleLocator(0.5))
    subplot.yaxis.set_major_locator(tick.MultipleLocator(0.1))
    subplot.yaxis.set_minor_locator(tick.MultipleLocator(0.05))
    plt.show()
    plt.grid(True)

def exp():
    fil = open("exp.txt", "r")
    exp = []
    i = 0
    for line in fil.readlines():
        for j in range(len(line.split())):
            exp.append(float(line.split()[j].replace(",", ".")))

    print(exp)
    exp.sort()
    print(exp)
    m = 1 + int(math.log2(200))
    print(m)
    a = [0]*(m + 1)
    a[0] = 0
    a[m] = max(exp)
    h = round((a[m] - a[0])/m, 5)
    for i in range(len(a)):
        if i + 1 != m and i != m:
            a[i + 1] = round(a[i] + h, 5)
    print(a)
    xi = [0] * m
    for i in range(len(xi)):
        if i + 1 != m+1:
            xi[i] = round((a[i] + a[i+1])/2, 5)
    print(xi)
    l = len(xi) - 2
    ni = [0] * m
    for i in range(len(a) - 1):
        for j in range(len(exp)):
            if exp[j] >= a[i] and exp[j] <= a[i + 1]:
                ni[i] += 1
    print(ni)
    print(sum(ni))
    wi = [0] * m
    for i in range(len(ni)):
        wi[i] = round(ni[i]/200, 5)
    print(wi)
    print(sum(wi))

    #-----------------------
    u1 = 0
    for i in range(len(xi)):
        u1 += round(xi[i] * wi[i], 5)
    lamba = round(1/u1, 5)
    f = [0] * (m + 1)
    for i in range(len(f)):
        f[i] = round(lamba * math.exp(-lamba * a[i]), 5)
    print(f)
    F = [0] * (m + 1)
    for i in range(len(F)):
        F[i] = round(1 - math.exp(-lamba * a[i]), 5)
    print(F)

    p = []
    for i in range(m+1):
        if i == 0:
            p.append(0)
        elif i == m:
            p.append(round(1 - F[i-1], 5))
        else:
            p.append(round(F[i] - F[i - 1], 5))
    print(p)
    print(sum(p))


    plt.axis([0, xi[len(xi) - 1] + 1, 0, 1.7])


    subplot = plt.subplot()
    subplot.tick_params(which='both', width=1)
    subplot.tick_params(which='major', length=7)
    subplot.minorticks_on()
    sub = plt.subplot()
    sub.tick_params(which='both', width=1)
    sub.tick_params(which='major', length=7)
    sub.tick_params(which='minor', length=4, color='red')
    sub.plot(a, f, linewidth=2.5, color="blue")
    ax = plt.subplot()
    ax.hist(exp, bins = a, density=True, color='white', edgecolor='black', linewidth=1)
    sub.minorticks_on()
    sub.xaxis.set_major_locator(tick.MultipleLocator(1))
    sub.xaxis.set_minor_locator(tick.MultipleLocator(0.5))
    sub.yaxis.set_major_locator(tick.MultipleLocator(0.1))
    sub.yaxis.set_minor_locator(tick.MultipleLocator(0.05))

    plt.show()
    plt.grid(True)

    #--------------
    wipi = []
    for j in range(len(wi)):
        wipi.append(round(abs(wi[j] - p[j+1]), 5))

    roundNpi = []
    for j in range(len(wipi)):
        roundNpi.append(round(200 * pow(wipi[j], 2) / p[j+1], 5))

    print(wipi)
    print(max(wipi))
    print(roundNpi)
    print(sum(roundNpi))

    Xkr = 0
    if l == 4:
        Xkr = 9.5
    elif l == 5:
        Xkr = 11.1
    elif l == 6:
        Xkr = 12.6
    elif l == 7:
        Xkr = 14.1
    elif l == 8:
        Xkr = 15.5
    print("Выборочное значение критерия Xkr^2", Xkr)
    if Xkr >= sum(roundNpi):
        print("Не противоречит")
    else:
        print("Противоречит")

def norm():
    fil = open("norm.txt", "r")
    norm = []
    i = 0
    for line in fil.readlines():
        for j in range(len(line.split())):
            norm.append(float(line.split()[j].replace(",", ".")))

    print(norm)
    norm.sort()
    print(norm)
    m = 1 + int(math.log2(200))
    print(m)
    a = [0] * (m + 1)
    a[0] = min(norm)
    a[m] = max(norm)
    h = round((a[m] - a[0]) / m, 5)
    for i in range(len(a)):
        if i + 1 != m and i != m:
            a[i + 1] = round(a[i] + h, 5)
    print(a)
    xi = [0] * m
    for i in range(len(xi)):
        if i + 1 != m + 1:
            xi[i] = round((a[i] + a[i + 1]) / 2, 5)
    print(xi)
    l = len(xi) - 3
    ni = [0] * m
    for i in range(len(a) - 1):
        for j in range(len(norm)):
            if norm[j] >= a[i] and norm[j] <= a[i + 1]:
                ni[i] += 1
    print(ni)
    print(sum(ni))
    wi = [0] * m
    for i in range(len(ni)):
        wi[i] = round(ni[i] / 200, 5)
    print(wi)
    print(sum(wi))

    vib_sr = 0
    for i in range(len(xi)):
        vib_sr += xi[i]*wi[i]

    vib_dis = 0
    for i in range(len(xi)):
        vib_dis += pow(xi[i] - vib_sr, 2) * wi[i]

    print("Выборочное среднее: ", round(vib_sr, 5))
    print("Выборочная дисперсия: ", round(vib_dis, 5))
    half_dis = round(math.sqrt(vib_dis), 5)
    print(half_dis)
    # -----------------------

    drob_a = [0] * (m + 1)
    for i in range(len(drob_a)):
        drob_a[i] = round((a[i] - vib_sr)/half_dis, 5)
    print(drob_a)

    f = [0] * (m + 1)
    for i in range(len(f)):
        f[i] = round((math.exp(-pow(drob_a[i], 2)/2)/math.sqrt(2*math.pi))/half_dis, 5)
    print(f)

    F = [0] * (m + 1)
    for i in range(len(F)):
        F[i] = round(scipy.stats.norm.cdf(drob_a[i]), 5)
    print(F)

    p = []
    for i in range(m + 1):
        if i == 0:
            p.append(0)
        elif i == 1:
            p.append(F[1])
        elif i == m:
            p.append(round(1 - F[i - 1], 5))
        else:
            p.append(round(F[i] - F[i - 1], 5))
    print(p)
    print(round(sum(p), 1))
    # -----------------------
    plt.axis([-4, xi[len(xi) - 1] + 1, 0, 1])

    subplot = plt.subplot()
    subplot.tick_params(which='both', width=1)
    subplot.tick_params(which='major', length=7)
    subplot.minorticks_on()
    sub = plt.subplot()
    sub.tick_params(which='both', width=1)
    sub.tick_params(which='major', length=7)
    sub.tick_params(which='minor', length=4, color='red')
    sub.plot(a, f, linewidth=2.5, color="blue")
    ax = plt.subplot()
    ax.hist(norm, bins=a, density=True, color='white', edgecolor='black', linewidth=1)
    sub.minorticks_on()
    sub.xaxis.set_major_locator(tick.MultipleLocator(1))
    sub.xaxis.set_minor_locator(tick.MultipleLocator(0.5))
    sub.yaxis.set_major_locator(tick.MultipleLocator(0.1))
    sub.yaxis.set_minor_locator(tick.MultipleLocator(0.05))

    plt.show()
    plt.grid(True)

    # --------------
    wipi = []
    for j in range(len(wi)):
        wipi.append(round(abs(wi[j] - p[j + 1]), 5))

    roundNpi = []
    for j in range(len(wipi)):
        roundNpi.append(round(200 * pow(wipi[j], 2) / p[j + 1], 5))

    print(wipi)
    print(max(wipi))
    print(roundNpi)
    print(round(sum(roundNpi), 5))

    Xkr = 0
    if l == 4:
        Xkr = 9.5
    elif l == 5:
        Xkr = 11.1
    elif l == 6:
        Xkr = 12.6
    elif l == 7:
        Xkr = 14.1
    elif l == 8:
        Xkr = 15.5
    print("Выборочное значение критерия Xkr^2", Xkr)
    if Xkr >= sum(roundNpi):
        print("Не противоречит")
    else:
        print("Противоречит")

def ravn():
    fil = open("ravn.txt", "r")
    ravn = []
    i = 0
    for line in fil.readlines():
        for j in range(len(line.split())):
            ravn.append(float(line.split()[j].replace(",", ".")))

    print(ravn)
    ravn.sort()
    print(ravn)
    m = 1 + int(math.log2(200))
    print(m)
    a = [0] * (m + 1)
    a[0] = A
    a[m] = B
    h = round((a[m] - a[0]) / m, 5)
    for i in range(len(a)):
        if i + 1 != m and i != m:
            a[i + 1] = round(a[i] + h, 5)
    print(a)
    xi = [0] * m
    for i in range(len(xi)):
        if i + 1 != m + 1:
            xi[i] = round((a[i] + a[i + 1]) / 2, 5)
    print(xi)
    l = len(xi) - 1
    ni = [0] * m
    for i in range(len(a) - 1):
        for j in range(len(ravn)):
            if ravn[j] >= a[i] and ravn[j] <= a[i + 1]:
                ni[i] += 1
    print(ni)
    print(sum(ni))
    wi = [0] * m
    for i in range(len(ni)):
        wi[i] = round(ni[i] / 200, 5)
    print(wi)
    print(sum(wi))
    # --------------
    p = []
    for i in range(m + 1):
        p.append(1/m)
    print(p)
    wipi = []
    for j in range(len(wi)):
        wipi.append(round(abs(wi[j] - p[j + 1]), 5))

    roundNpi = []
    for j in range(len(wipi)):
        roundNpi.append(round(200 * pow(wipi[j], 2) / p[j + 1], 5))

    print(wipi)
    print(max(wipi))
    print(roundNpi)
    print(round(sum(roundNpi), 5))
    # -----------------------
    f = []
    for i in range(m+1):
        f.append(round(1/(B-A), 5))
    print(f)

    plt.axis([4.2, xi[len(xi) - 1] + 1, 0, 0.2])

    subplot = plt.subplot()
    subplot.tick_params(which='both', width=1)
    subplot.tick_params(which='major', length=7)
    subplot.minorticks_on()
    sub = plt.subplot()
    sub.tick_params(which='both', width=1)
    sub.tick_params(which='major', length=7)
    sub.tick_params(which='minor', length=4, color='red')
    sub.plot(a, f, linewidth=2.5, color="blue")
    ax = plt.subplot()
    ax.hist(ravn, bins=a, density=True, color='white', edgecolor='black', linewidth=1)
    sub.minorticks_on()
    sub.xaxis.set_major_locator(tick.MultipleLocator(1))
    sub.xaxis.set_minor_locator(tick.MultipleLocator(0.5))
    sub.yaxis.set_major_locator(tick.MultipleLocator(0.1))
    sub.yaxis.set_minor_locator(tick.MultipleLocator(0.05))

    plt.show()
    plt.grid(True)

    # --------------
    Xkr = 0
    if l == 4:
        Xkr = 9.5
    elif l == 5:
        Xkr = 11.1
    elif l == 6:
        Xkr = 12.6
    elif l == 7:
        Xkr = 14.1
    elif l == 8:
        Xkr = 15.5
    print("Выборочное значение критерия Xkr^2", Xkr)
    if Xkr >= sum(roundNpi):
        print("Не противоречит")
    else:
        print("Противоречит")
def kal():
    fil = open("ravn.txt", "r")
    kal = []
    i = 0
    for line in fil.readlines():
        for j in range(len(line.split())):
            kal.append(float(line.split()[j].replace(",", ".")))

    print(kal)
    kal.sort()
    print(kal)


#binom()
#exp()
norm()
#ravn()
#kal()