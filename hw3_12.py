import argparse
import math

import numpy as np
#  python .\hw3_12.py -m_1 0 -v_1 1 -w 1 2 3 4 5 -n 4 -v_2 2 -m_2 3 -s 5

def normal_dist(mean, std_dev):
    u, v = np.random.random_sample(2)
    e = (np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v))*std_dev + mean # Box-Muller method

    return e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 1-1
    parser.add_argument("-m_1", "--mean_1", type=int)
    parser.add_argument("-v_1", "--var_1", type=int)
    # 1-2
    parser.add_argument("-w", "--weights", type=float, nargs="+")
    parser.add_argument("-n", "--base", type=int)
    parser.add_argument("-v_2", "--var_2", type=int, help="Variance of Polynomial basis linear model data generator")
    # 2
    parser.add_argument("-m_2", "--mean_2", type=int)
    parser.add_argument("-s", "--s_var", type=int)

    args = parser.parse_args()

    # 1-a 
    mean = args.mean_1
    std_dev = np.sqrt(args.var_1)
    print ("Normal Dist Data point: ", normal_dist(mean, std_dev))

    # 1-b
    weights = [float(x) for x in args.weights]
    X = 2 * np.random.random_sample(len(weights)) - 1 # fit to range(-1, 1)
    
    e = normal_dist(0, np.sqrt(args.var_2))
    for w, x in zip(weights, X):
        e += w*x
    print("Polynomial Basis value: ", e)

    # 2
    sim_mean, sim_var, num_datum, M2 = 0, 0, 0, 0
    while abs(sim_mean - args.mean_2) > 0.01 or abs(sim_var - args.s_var) > 0.01:
        val = normal_dist(args.mean_2, np.sqrt(args.s_var))
        num_datum += 1
        delta1 = val - sim_mean
        sim_mean += delta1 / num_datum
        delta2 = val - sim_mean
        M2 += delta1 * delta2
        sim_var = M2 / num_datum
        print (f"Add data point: {val}")
        print (f"Mean = {sim_mean}  Variance = {sim_var}\n")

    print (f"Final Mean = {sim_mean}, Variance = {sim_var}, {num_datum} data points in total")

    return