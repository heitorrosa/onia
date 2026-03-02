# Calculo de peso usado na questao 15

import numpy as np

x = np.array([1, 2, 1, 3, 0])
w = np.array([2, 1, -3, -2, 4])
bias = 2

def relu(sum):
    return max(0, sum)

if __name__ == "__main__":
    sigma = np.sum(x * w)
    sigma += bias

    print(f'Σ: {sigma}')
    print(f'σ: {relu(sigma)}')