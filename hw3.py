import argparse
import math

import numpy as np
#  python .\hw3.py -m_1 0 -v_1 1 -w 1 2 3 4 5 -n 4 -v_2 2 -m_2 3 -s 5

def normal_dist(mean, std_dev):
    u, v = np.random.random_sample(2)
    e = (np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v))*std_dev + mean # Box-Muller method

    return e


def polynomial_dist(weights, X, e):
    ans, power = e, 1
    for w, x in zip(weights, X):
        ans += w*x**power
        power += 1

    return ans


def plot_info(x, y, mean, variance, pd_mean, pd_var):
    print (f"Add data point ({x[0]}, {y}):\n")
    print (f"Posterior mean:\n")
    print (mean, "\n")
    print ("Poseterior variance:\n")
    print (variance)
    print (f"\nPredictive Distribution ~ N({pd_mean}, {pd_var})\n")
    print ("-----------------------")

    return


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
    # 3
    parser.add_argument("-p", "--precision", type=int)

    args = parser.parse_args()

    # 1-a 
    mean = args.mean_1
    std_dev = np.sqrt(args.var_1)
    print ("Normal Dist Data point: ", normal_dist(mean, std_dev))

    # 1-b
    weights = [float(x) for x in args.weights]
    X = 2 * np.random.random_sample(1) - 1 # fit to range(-1, 1)
    e = normal_dist(0, np.sqrt(args.var_2))
    print("Polynomial Basis value: ", polynomial_dist(weights, X, e))

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
        # print (f"Add data point: {val}")
        # print (f"Mean = {sim_mean}  Variance = {sim_var}\n")

    print (f"Final Mean = {sim_mean}, Variance = {sim_var}, {num_datum} data points in total\n")

    # 3
    b = args.precision
    a = args.var_2
    w_posteriror_mean = np.zeros((len(weights)))
    w_posteriror_cov = np.identity(len(weights))*b
    weights = np.array(weights)
    first = True

    while abs(np.sum(w_posteriror_mean - weights)) > 0.1*len(weights):
        data_x = 2 * np.random.random_sample(1) - 1
        e = normal_dist(0, np.sqrt(a))
        data_y = polynomial_dist(weights, data_x, e)
        
        if first:
            X = np.array([data_x[0]**i for i in range(len(weights))])
            y = np.array([data_y])
            print (X)
            print (y, "\n")

            w_posteriror_cov += a * np.dot(np.transpose(X), X)
            w_posteriror_mean = a * np.dot(np.linalg.inv(w_posteriror_cov), np.transpose(X))
            first = False
            plot_info(data_x, data_y, w_posteriror_mean, w_posteriror_cov, 0, 0)
        else:
            X = np.append(X, np.array([[data_x**i for i in range(len(weights))]]))

        # w_posteriror_mean = 
    
