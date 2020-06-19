import csv
import pandas as pd
import matplotlib.pyplot as plt
import random as rn

# TODO: a의 최적해를 찾아야 한다.
# 데이터 셋 찾음
# MSE 오차율을 유전알고리즘으로 구하기
# 기울기 음수인 y = -ax 그래프. 
# a를 구하기위한 랜덤값 x와 y 
# x와 y를 가지고 원래 데이터와 비교하여 오차율 계산
# 적은 계산값을 가지고 교차연산
# MSE 그래프 가로축 기울기, 세로축은 오차율
# TODO: 기존 유전알고리즘에서 초기식 y = -ax로 설정
# TODO: 랜덤 a에서 후보해 4개 선정하기
# TODO: 작은값 선택하기

# ----------------유전알고리즘으로 기울기 a의 최적해 찾기-----------------


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
