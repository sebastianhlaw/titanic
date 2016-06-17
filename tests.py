
import numpy as np

arr = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
print(arr)

rows, cols = arr.shape

for x in range(0, cols):
    subarr = [i for i in range(0, cols)]
    del subarr[x]
    arr[:, subarr]
