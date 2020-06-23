import csv
import pandas as pd
import matplotlib.pyplot as plt
import random as rn
import math

# Kaggle에서 Tesla의 주식 데이터를 사용하였다.
# https://www.kaggle.com/rpaguirre/tesla-stock-price

# 데이터 셋 찾음
# MSE 오차율을 유전알고리즘으로 구하기
# 기울기 음수인 y = ax 그래프.
# a를 구하기위한 랜덤값 x와 y
# x와 y를 가지고 원래 데이터와 비교하여 오차율(MSE) 계산
# 오차율이 가장 작은 값을 구하기위해 기울기에 따른 오차율 유전알고리즘
# MSE 그래프 가로축 기울기, 세로축은 오차율

# ----------------주어진 데이터셋--------------------
f = open('Tesla.csv', 'r')
rdr = csv.reader(f)

arr = []
arr2 = []
arr3 = []

for line in rdr:
    # Open
    arr.append(float(line[1]))
    # High
    arr2.append(float(line[2]))
    # Low
    arr3.append(float(line[3]))

# High와 Low의 평균값
avg = []
for i in range(len(arr2)):
    av = (arr2[i] + arr3[i]) / 2
    avg.append(float(av))


# ----------------유전알고리즘으로 기울기 a의 최적해 찾기-----------------
# 초기 식 y = ax
def expression(t, h):
    return h / t


# 예측된 값으로 그래프 그리기
def predicted(a, xVal):
    yArr = []
    for i in range(len(xVal)):
        if a < 0:
            y = -a * xVal[i]
        else:
            y = a * xVal[i]
        yArr.append(float(y))
    return yArr


# 초기 a 4개 구하기
def init(Open1, Open2, Avg1, Avg2):
    a = []
    for i in range(4):
        OPEN = rn.randint(Open1, Open2)
        AVG = rn.randint(Avg1, Avg2)

        a.append(math.ceil(expression(OPEN, AVG)))
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


# bin() format
def binformat(x):
    binformat = '{0:>8}'.format(bin(x)).replace("b", "0").replace(" ", "0")
    return binformat


# int to binary
def int2Bin(str):
    binlist = []
    for i in range(4):
        binsrting = binformat(str[i])
        binlist.append(binsrting)
    return binlist


# 교차 연산
def crossover(arr):
    strarr = []
    for i in range(2):
        bita = str(arr[i])
        bitb = str(arr[i + 1])

        strarr.append(bita[:5] + bitb[5:])
        strarr.append(bitb[:5] + bita[5:])

    return strarr


# 돌연변이 연산
def invert(char):
    ran = rn.random()
    a = int(char, 2)  # 숫자
    b = int("1000", 2)

    for i in range(5):
        p = 1 / 30
        if ran < p:
            if char[4:5] == "1":
                aa = a - b

            elif char[4:5] == "0":
                aa = a + b
            return aa
        return a


# 돌연변이 한 값 저장하기
def mutation(mut):
    mutarr = []
    mutint = list(map(str, mut))
    for i in range(4):
        mutinv = float(invert(mutint[i]))
        mutarr.append(mutinv)
    return mutarr


# MSE 계산식
def MSEexpression(yInit, yPredict):
    MSEval = yPredict - yInit
    powMSE = MSEval * MSEval
    return powMSE


# 오차율 구하기
def MSE(predict, tempdata):
    initgraph = []
    yYval = []
    for i in range(4):
        for j in range(1000):
            initnotation = predict[i] * tempdata[j]
            yY = MSEexpression(arr2[j], initnotation)
            yYval.append(float(yY))
            yMSE = 1 / 150 * sum(yYval)
        initgraph.append(float(yMSE))
    return initgraph


# 기울기와 MSE값을 가진 리스트에서 최소 MSE의 기울기를 찾아낸다.
def minx(x, list2):
    for lis in list2:
        if str(x) in lis:
            return float(lis[0])


# 유전알고리즘 ---------main---------
Openmin = 2
Openmax = 300
Avgmin = 2
Avgmax = 300

x = []
y = []
list2 = []
RanOpenprice = []

# 반복 100번
for i in range(100):
    # 각 함수들의 사용
    initreturn = init(Openmin, Openmax, Avgmin, Avgmax)
    selectionreturn = selection(initreturn)
    inttobinary = int2Bin(selectionreturn)
    cross = crossover(inttobinary)
    mutationed = mutation(cross)
    MSEarr = MSE(mutationed, arr)

    # print를 통해 원하는 값이 나오는지 확인
    print("---------------------")
    print("init: ", initreturn)
    print("selection:", selectionreturn)
    print("inttobin: ", inttobinary)
    print("cross: ", cross)
    print("mutationed: ", mutationed)
    print("MSEarr:", MSEarr)
    print("예측된 기울기:", min(mutationed))
    print("-----------------------")

    # x와 y에 예측된 기울기와 MSE값을 넣은다.
    x += mutationed
    y += MSEarr

    # 예측된 기울기 값과 랜덤 온도로 그래프 예측하기위한 값
    Open_Price = rn.randint(2, 300)
    RanOpenprice.append(Open_Price)

    # x와 y에 예측된 기울기와 MSE값을 넣은다.
    for u in range(len(y)):
        list2.append([str(x[i]), str(y[i])])

# minx 함수에서 최소 MSE의 기울기를 알아낸다.
minX = minx(min(y), list2)
print("최소 MSE 기울기: ", minX)

# 그래프 예측을 위한 계산(predicted 함수 사용)
yPredict = predicted(minX, RanOpenprice)

# pandas library로 DataFrame 지정
MSEGrapg = pd.DataFrame(
    {"aValue": x, "MSE": y}
)

OpPrice = pd.DataFrame(
    {"temp": arr, "hum": avg}
)

predictgraph = pd.DataFrame(
    {"temp2": RanOpenprice, "hum2": yPredict}
)

# 그래프 그리기
plt.figure()
# 기존 데이터의 그래프를 그린다
one = plt.scatter(OpPrice['temp'], OpPrice['hum'], marker="o")
# 예측된 기울기로 그래프를 예측한다.
two = plt.scatter(predictgraph["temp2"], predictgraph["hum2"], marker="x")
plt.legend(handles=(one, two), labels=("Initial data", "predicted data"), loc="upper left")
plt.xlabel('Open_Price')
plt.ylabel('Average_Price')
plt.savefig("Tesla.png")

plt.figure()
# MSE 평가지표 그래프
MSEleg = plt.scatter(MSEGrapg['aValue'], MSEGrapg['MSE'], marker="o")
plt.xlabel('Inclination')
plt.ylabel('MSE')
plt.savefig("TeslaMSE.png")

# 그래프 띄우기
plt.show()
