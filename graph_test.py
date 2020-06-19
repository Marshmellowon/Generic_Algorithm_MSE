import matplotlib.pyplot as plt
import numpy as np

x = np.array([2.3, 2.5, 3, 4.6, 5, 6.9])
y = np.array([4, 5, 6.8, 7, 8.5, 4.5])

print(type(x[1]))
plt.figure()
plt.scatter(x, y)
plt.xlabel('temperature')
plt.ylabel('humedity')
plt.show()
