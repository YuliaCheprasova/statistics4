import random
import numpy as np
import math as m
import matplotlib.pyplot as plt
from openpyxl import load_workbook


def Y(x):
    y=np.zeros(len(x))
    k = random.uniform(-10.01, 10.01)
    b = random.uniform(10.99, 1.01)
    c = random.uniform(-10.01, 10.01)
    """c=1
    k=10
    b=50"""
    print( k, b, c)
    for i in range(len(x)):
        h = np.random.normal(0, 100.5, 1)[0]
        y[i]=(k*(x[i]-av(x)) + b*(x[i])**2 + c) + h
    return y


def av(x):
    sum=0
    for i in x:
       sum+=i
    return sum/len(x)


if __name__ == '__main__':
    n=1000
    x = np.zeros(n)
    for i in range(1,n+1):
        x[i-1]=i*0.1
    y = Y(x)

    #plt.plot(x, y)
    #plt.show()
    """xav=av(x)
    yav=av(y)
    xy = np.zeros(n)
    for i in range(n):
        xy[i] = x[i]*y[i]
    xyav=av(xy)
    xsq=np.square(x)
    xsqav=av(xsq)
    xavsq = np.square(xav)
    kpr = (xyav-xav*yav)/(xsqav-xavsq)
    bpr = yav-kpr*xav
    """

    """xy = np.zeros(n)
    for i in range(n):
        xy[i] = x[i] * y[i]
    xsq = np.square(x)
    kpr = sum(xy)/sum(xsq)"""
    """sinxy = np.zeros(n)
    sinxsq = np.zeros(n)
    for i in range(n):
        sinxy[i] = m.sin(x[i])*y[i]
        sinxsq[i] = (m.sin(x[i]))**2
    kpr = sum(sinxy)/sum(sinxsq)"""



    """res = np.zeros((3,3))
    xav = av(x)
    x_av = np.zeros(n)
    x_avsq = np.zeros(n)
    x_avx = np.zeros(n)
    x_avy = np.zeros(n)
    for i in range(n):
        x_av[i] = x[i]-xav
        x_avy[i] = (x[i] - xav) * y[i]
        x_avsq[i] = (x[i]-xav)**2
        x_avx[i] = (x[i]-xav)*(x[i])**2
    res[0][0] = sum(x_avsq)
    res[0][1] = sum(x_avx)
    res[1][0] = sum(x_avx)
    res[0][2] = sum(x_av)
    res[2][0] = sum(x_av)
    #kpr = sum(x_av)/sum(x_avsq)
    x_2 = np.zeros(n)
    x_2y = np.zeros(n)
    for i in range(n):
        x_2[i] = (x[i])**2
        x_2y[i] = y[i] * (x[i])**2
    x_4 = np.square(x_2)
    res[1][1] = sum(x_4)
    res[1][2] = sum(x_2)
    res[2][1] = sum(x_2)
    res[2][2] = n
    b = np.zeros(3)
    b[0] = sum(x_avy)
    b[1] = sum(x_2y)
    b[2] = sum(y)
    coef = np.linalg.solve(res, b)
    kpr = coef[0]
    bpr = coef[1]
    cpr = coef[2]
    #bpr = sum(x_2y)/sum(x_4)
    #cpr = sum(y)/n
    print(kpr, bpr, cpr)
    ypredict = np.zeros(n)
    for i in range(n):
        ypredict[i] = kpr*(x[i]-xav) + bpr*(x[i])**2 + cpr
    plt.plot(x, ypredict)
    plt.show()
    E = np.zeros(n)
    for j in range(n):
        E[j] = (ypredict[j] - y[j]) ** 2
    print(sum(E))"""

    n2 = 200
    square = np.zeros(n2)
    price = np.zeros(n2)
    i = 0
    wb = load_workbook('./selection.xlsx')
    sheet = wb['cian_parsing_result_sale_1_10_k']
    for cellObj in sheet['I2':'I201']:
        for cell in cellObj:
            square[i] = cell.value
            i += 1

    i = 0
    for cellObj in sheet['J2':'J201']:
        for cell in cellObj:
            price[i] = cell.value
            i += 1


    plt.plot(square, price)
    # plt.show()

    res2 = np.zeros((3,3))
    xav2 = av(square)
    x_av2 = np.zeros(n2)
    x_avsq2 = np.zeros(n2)
    x_avx2 = np.zeros(n2)
    x_avy2 = np.zeros(n2)
    for i in range(n2):
        x_av2[i] = square[i]-xav2
        x_avy2[i] = (square[i] - xav2) * price[i]
        x_avsq2[i] = (square[i]-xav2)**2
        x_avx2[i] = (square[i]-xav2)*(square[i])**2
    res2[0][0] = sum(x_avsq2)
    res2[0][1] = sum(x_avx2)
    res2[1][0] = sum(x_avx2)
    res2[0][2] = sum(x_av2)
    res2[2][0] = sum(x_av2)
    #kpr = sum(x_av)/sum(x_avsq)
    x_22 = np.zeros(n2)
    x_2y2 = np.zeros(n2)
    for i in range(n2):
        x_22[i] = (square[i])**2
        x_2y2[i] = price[i] * (square[i])**2
    x_42 = np.square(x_22)
    res2[1][1] = sum(x_42)
    res2[1][2] = sum(x_22)
    res2[2][1] = sum(x_22)
    res2[2][2] = n2
    b2 = np.zeros(3)
    b2[0] = sum(x_avy2)
    b2[1] = sum(x_2y2)
    b2[2] = sum(price)
    coef2 = np.linalg.solve(res2, b2)
    kpr2 = coef2[0]
    bpr2 = coef2[1]
    cpr2 = coef2[2]
    #bpr = sum(x_2y)/sum(x_4)
    #cpr = sum(y)/n
    print(kpr2, bpr2, cpr2)
    ypredict2 = np.zeros(n2)
    for i in range(n2):
        ypredict2[i] = kpr2*(square[i]-xav2) + bpr2*(square[i])**2 + cpr2
    plt.plot(square, ypredict2)
    plt.show()
    E2 = np.zeros(n2)
    for j in range(n2):
        E2[j] = (ypredict2[j] - price[j]) ** 2
    print(sum(E2))

