import csv
import pandas as pd
import matplotlib.pyplot as plt
import random as rn
import numpy as np

# 데이터 셋 찾음
# MSE 오차율을 유전알고리즘으로 구하기
# 기울기 음수인 y = -ax 그래프. 
# a를 구하기위한 랜덤값 x와 y 
# x와 y를 가지고 원래 데이터와 비교하여 오차율(MSE) 계산
# 오차율이 가장 작은 값을 구하기위해 기울기에 따른 오차율 유전알고리즘

# MSE 그래프 가로축 기울기, 세로축은 오차율

# ----------------주어진 데이터셋--------------------
f = open('temp.csv', 'r')
rdr = csv.reader(f)

arr = []
arr2 = []

for line in rdr:
    # 기온
    arr.append(float(line[3]))
    # 습도
    arr2.append(float(line[4]))


# ----------------유전알고리즘으로 기울기 a의 최적해 찾기-----------------
# 초기 식 y = -ax
def nota(t, h):
    return -h / t


# 예측된 값으로 그래프 그리기
def predicted(a, xVal):
    yArr = []
    print("a", a)
    print("xVal", xVal)
    for i in range(len(xVal)):
        y = (a - 0.5) * xVal[i] +50
        yArr.append(float(y))
    return yArr


# 초기 a 4개 구하기
def init(tmin, tmax, hmin, hmax):
    a = []
    for i in range(4):
        temp = rn.randint(tmin, tmax)
        hum = rn.randint(hmin, hmax)

        a.append(round(nota(temp, hum)))
    return a


# 선택 연산
def selection(Aarr):
    ratio = []
    for i in range(4):
        if i == 0:
            ratio.append(Aarr[0] / sum(Aarr))
        else:
            ratio.append(ratio[i - 1] + Aarr[i] / sum(Aarr))

    sx = []
    for i in range(4):
        p = rn.random()
        if p < ratio[0]:
            sx.append(Aarr[0])
        elif p < ratio[1]:
            sx.append(Aarr[1])
        elif p < ratio[2]:
            sx.append(Aarr[2])
        else:
            sx.append(Aarr[3])
    return sx


def binformat(x):
    binformat = '{0:>8}'.format(bin(x)).replace("b", "0").replace(" ", "0").replace("-", "0")
    return binformat


def int2Bin(str):
    binlist = []
    for i in range(4):
        binsrting = binformat(str[i])
        binlist.append(binsrting)
    return binlist


def crossover(arr):
    strarr = []
    for i in range(2):
        bita = str(arr[i])
        bitb = str(arr[i + 1])

        strarr.append(bita[:4] + bitb[4:])
        strarr.append(bitb[:4] + bita[4:])

    return strarr


def invert(char):
    ran = rn.random()
    a = int(char, 2)  # 숫자
    abin = binformat(a)
    abinstr = str(abin)
    reta = 0

    for i in range(5):
        p = 1 / 32
        if ran < p:
            if abinstr[5:6] == "1":
                reta = a & ~(2 << 3)

            elif abinstr[5:6] == "0":
                reta = a | (2 << 3)
            return reta
        return a


def mutation(mut):
    mutarr = []
    mutarr2 = []
    output = []
    mutint = list(map(str, mut))
    for i in range(4):
        mutarr.append(float(-invert(mutint[i])))

    return mutarr


# MSE 계산식
def MSEnotation(yInit, yPredict):
    MSEval = pow(yPredict - yInit, 2)
    return MSEval


# 오차율 구하기
def MSE(predict, tempdata):
    initgraph = []
    yYval = []
    for i in range(4):
        for j in range(144):
            initnotation = predict[i] * tempdata[j]
            yY = MSEnotation(arr2[i], initnotation)
            yYval.append(float(yY))
            yMSE = 1 / 2000000 * sum(yYval)
        initgraph.append(float(yMSE))

    return initgraph


def minx(x, list2):
    for lis in list2:
        if str(x) in lis:
            return float(lis[0])


# TODO: 선정된 4개의 a와 랜덤한 기온값 대입 vs 초기 데이터와 비교(MSE) MSE그래프 그리기
# TODO: 최소 오차율인 기울기로 랜덤한 기온값 대입하여 예측그래프 그리기
# MSE = 1/n sum(Y'-Y)^2
# 유전알고리즘 main---------
tempmin = 5
tempmax = 35
hummin = 10
hummax = 95

x = []
y = []
list2 = []
TempPred = []

for i in range(100):
    initreturn = init(tempmin, tempmax, hummin, hummax)
    selectionreturn = selection(initreturn)
    inttobinary = int2Bin(selectionreturn)
    cross = crossover(inttobinary)
    mutationed = mutation(cross)
    MSEarr = MSE(mutationed, arr)

    # print test
    print("---------------------")
    print("init: ", initreturn)
    print("selection:", selectionreturn)
    print("inttobin: ", inttobinary)
    print("cross: ", cross)
    print(mutationed)
    print(MSEarr)

    x += mutationed
    y += MSEarr

    list2.append([str(x[i]), str(y[i])])

    temperature = rn.randint(tempmin, tempmax)
    TempPred.append(float(temperature))

    minX = minx(min(y), list2)
    if minX == None:
        print("none")

print("prit", minX)
yPredict = predicted(minX, TempPred)

MSEGrapg = pd.DataFrame(
    {"aValue": x, "MSE": y}
)

temp = pd.DataFrame(
    {"temp": arr, "hum": arr2}
)
predictgraph = pd.DataFrame(
    {"temp2": TempPred, "hum2": yPredict}
)

print(list2)

plt.figure()
plt.subplot(2, 1, 2)
plt.scatter(temp['temp'], temp['hum'], marker="o")
plt.scatter(predictgraph["temp2"], predictgraph["hum2"], marker="x")
plt.xlabel('temperature')
plt.ylabel('humedity')

plt.subplot(2, 2, 2)
plt.scatter(MSEGrapg['aValue'], MSEGrapg['MSE'], marker="o")
plt.xlabel('temperature')
plt.ylabel('humedity')
plt.show()

breakpoint()
# #####################################################################


temp = pd.DataFrame(
    {"temp": arr, "hum": arr2}
)
# ########################################################################

# -----------------그래프 그리기----------------------
plt.figure()

plt.show()
