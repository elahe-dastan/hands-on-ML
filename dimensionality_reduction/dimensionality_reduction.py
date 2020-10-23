import numpy as np

X = np.array([[3, 2, 2],
              [2, 3, -2]])

X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]

print(U)
print(s)
print(Vt)