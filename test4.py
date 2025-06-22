import numpy as np

a = np.array([1, 2])
b = np.array([[1, 2, 3], [4, 5, 6]])

try:
    c = np.stack(a, b, axis=0)
except Exception as e:
    print(e)