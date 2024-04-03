def linearCongruentialMethod(Xo, m, a, c,
                             randomNums,
                             noOfRandomNums):
    # Initialize the seed state
    randomNums[0] = Xo

    # Traverse to generate required
    # numbers of random numbers
    for i in range(1, noOfRandomNums):
        # Follow the linear congruential method
        randomNums[i] = ((randomNums[i - 1] * a) +
                         c) % m


import math

import numpy as np


def blum_blum_shub(p, q, seed, length):
    M = p * q
    x = seed
    bits = []
    for _ in range(length):
        x = pow(x, 2, M)  # x_n+1 = x_n^2 mod M
        bits.append(x % 2)  # Take the least significant bit
    return bits


def transform_vector_to_binary(vector):
    # Transform each element in the vector to 1 if it is not 0, otherwise to 0
    return [1 if num != 0 else 0 for num in vector]


def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


if __name__ == '__main__':
    Xo = 5  # Seed value
    m = 7  # Modulus parameter
    a = 3  # Multiplier term
    c = 3  # Increment term

    # Number of Random numbers to be generated
    noOfRandomNums = 50

    # To store random numbers
    randomNums = [0] * (noOfRandomNums)

    # Function Call
    linearCongruentialMethod(Xo, m, a, c,
                             randomNums,
                             noOfRandomNums)

    # Print the generated random numbers
    binary_vector = transform_vector_to_binary(randomNums)
    print(randomNums)
    print('\n')
    # Example parameters (In practice, p and q should be large primes for security)
    p = 11  # Example small prime, not secure
    q = 19  # Example small prime, not secure
    seed = 3  # Example seed
    length = 50  # Number of bits to generate

    bits = blum_blum_shub(p, q, seed, length)
    print(bits)

    MSE = np.square(np.subtract(binary_vector, bits)).mean()

    RMSE = math.sqrt(MSE)
    print("Root Mean Square Error:")
    print(RMSE)

    range_to_normalize = (0, 1)
    normalized_array_1 = normalize(binary_vector,
                                    range_to_normalize[0],
                                    range_to_normalize[1])
    range_to_normalize = (-1, 1)
    normalized_array_2 = normalize(bits,
                                    range_to_normalize[0],
                                    range_to_normalize[1])

    print("Normalize 1: ")
    print(normalized_array_1)
    print("Normalize 2: ")
    print(normalized_array_2)


