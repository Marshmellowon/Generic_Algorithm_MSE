import csv
import pandas as pd
import matplotlib.pyplot as plt

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
plt.figure()

plt.scatter(temp['temp'], temp['hum'], marker="o")
plt.xlabel('temperature')
plt.ylabel('humedity')
plt.show()
