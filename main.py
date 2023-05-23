import random
import numpy as np
import math as m
import matplotlib.pyplot as plt
from openpyxl import load_workbook


def Y(x):
    y=np.zeros(len(x))
    k = random.uniform(-5, 5)
    b = random.uniform(-5, 5)
    c = random.uniform(-5, 5)
    """c=1
    k=10
    b=50"""
    print(c, k, b)
    for i in range(len(x)):
        h = np.random.normal(0, 0.5, 1)[0]
        y[i]=(k*m.sin(x[i]) + b*(x[i])**2 + c) + h
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

    plt.plot(x, y)
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
        kpr = sum(sinxy)/sum(sinxsq)
    x_2 = np.zeros(n)
    x_2y = np.zeros(n)
    for i in range(n):
        x_2[i] = (x[i])**2
        x_2y[i] = y[i] * (x[i])**2
    x_4 = np.square(x_2)
    bpr = sum(x_2y)/sum(x_4)
    cpr = sum(y)/n
    print(kpr, bpr, cpr)
    ypredict = np.zeros(n)
    for i in range(n):
        ypredict[i] = kpr*m.sin(x[i]) + bpr*(x[i])**2 + cpr
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

    sinxy2 = np.zeros(n2)
    sinxsq2 = np.zeros(n2)
    for i in range(n2):
        sinxy2[i] = m.sin(square[i]) * price[i]
        sinxsq2[i] = (m.sin(square[i])) ** 2
    kpr2 = sum(sinxy2) / sum(sinxsq2)
    x_22 = np.zeros(n2)
    x_2y2 = np.zeros(n2)
    for i in range(n2):
        x_22[i] = (square[i]) ** 2
        x_2y2[i] = price[i] * (square[i]) ** 2
    x_42 = np.square(x_22)
    bpr2 = sum(x_2y2) / sum(x_42)
    cpr2 = sum(price) / n2
    print(kpr2, bpr2, cpr2)
    ypredict2 = np.zeros(n2)
    for i in range(n2):
        ypredict2[i] = kpr2 * m.sin(x[i]) + bpr2 * (x[i]) ** 2 + cpr2
    plt.plot(square, ypredict2)
    plt.show()
    E2 = np.zeros(n2)
    for j in range(n2):
        E2[j] = (ypredict2[j] - price[j]) ** 2
    print(sum(E2))

