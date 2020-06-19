import csv
import pandas as pd
import matplotlib.pyplot as plt
import random as rn


# 데이터 셋 찾음
# MSE 오차율을 유전알고리즘으로 구하기
# 기울기 음수인 y = -ax 그래프. 
# a를 구하기위한 랜덤값 x와 y 
# x와 y를 가지고 원래 데이터와 비교하여 오차율(MSE) 계산
# 오차율이 가장 작은 값을 구하기위해 기울기에 따른 오차율 유전알고리즘

# MSE 그래프 가로축 기울기, 세로축은 오차율

# ----------------유전알고리즘으로 기울기 a의 최적해 찾기-----------------
# 초기 식 y = -ax
def nota(t, h):
    return -h / t


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

    for i in range(5):
        p = 1 / 32
        if ran < p:
            reta = 0
            if abinstr[6:7] == "1":
                reta = a & ~(1 << 1)

            elif abinstr[6:7] == "0":
                reta = a | (1 << 1)
            return reta
        return a


def mutation(mut):
    mutarr = []
    mutarr2 = []
    output = []
    mutint = list(map(str, mut))
    for i in range(4):
        mutarr.append(-invert(mutint[i]))

    return mutarr


# TODO: 선정된 4개의 a와 랜덤한 기온값 대입 vs 초기 데이터와 비교(MSE) MSE그래프 그리기
# TODO: 최소 오차율인 기울기로 랜덤한 기온값 대입하여 예측그래프 그리기
# 유전알고리즘 main---------
tempmin = 5
tempmax = 35
hummin = 10
hummax = 95

initreturn = init(tempmin, tempmax, hummin, hummax)
selectionreturn = selection(initreturn)
inttobinary = int2Bin(selectionreturn)
cross = crossover(inttobinary)
mutationed = mutation(cross)

# print test
print(initreturn)
print(selectionreturn)
print(inttobinary)
print(cross)
print(mutationed)

breakpoint()
# #####################################################################

# ----------------주어진 데이터셋--------------------
f = open('temp.csv', 'r')
rdr = csv.reader(f)

arr = []
arr2 = []

for line in rdr:
    arr.append(float(line[3]))
    arr2.append(float(line[4]))

temp = pd.DataFrame(
    {"temp": arr, "hum": arr2}
)
# ########################################################################

# -----------------그래프 그리기----------------------
plt.figure()

plt.scatter(temp['temp'], temp['hum'], marker="o")
plt.xlabel('temperature')
plt.ylabel('humedity')
plt.show()
