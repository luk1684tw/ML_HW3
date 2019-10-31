import argparse
import math

import numpy as np
import matplotlib.pyplot as plt
#  python .\hw3.py -m_1 0 -v_1 1 -w 1 2 3 4 -n 4 -v_2 2 -m_2 3 -s 5 -p 1

def normal_dist(mean, std_dev):
    u, v = np.random.random_sample(2)
    e = (np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v))*std_dev + mean # Box-Muller method

    return e


def polynomial_dist(weights, x, e):
    ans = e
    for i, w in enumerate(weights):
        ans += w*x**i

    return ans


def plot_image(mode, weights, e, a, predict_results, w_posteriror_cov=None):
    plot_x = np.linspace(-2, 2, 100)
    plt.plot(plot_x, polynomial_dist(weights, plot_x, e), 'k')

    if mode != "running":
        plt.plot(plot_x, polynomial_dist(weights, plot_x, e) + np.sqrt(a), 'r')
        plt.plot(plot_x, polynomial_dist(weights, plot_x, e) - np.sqrt(a), 'r')
        plt.savefig("ground.png")
    
    plt.scatter(predict_results[:, 0], predict_results[:, 1])

    if mode != "running":
        plt.savefig("predict.png")
    else:
        y_var = list()
        for x in plot_x:
            design_x = np.array([x**i for i in range(len(weights))])
            y_var.append(1 / a + np.dot(np.dot(design_x, np.linalg.inv(w_posteriror_cov)), np.transpose(design_x)))

        y_var = np.array(y_var)

        y_predict = np.array([polynomial_dist(weights, x, e) for x in plot_x])
        plt.plot(plot_x, y_predict + y_var, 'r')
        plt.plot(plot_x, y_predict - y_var, 'r')
        plt.savefig(f"NO{len(predict_results)}.png")

    plt.close("all")
    return

def print_info(x, y, mean, variance, pd_mean, pd_var):
    print (f"Add data point ({x}, {y}):\n")
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
    w_posteriror_mean = np.zeros((len(weights), 1))
    w_prior_mean = np.copy(w_posteriror_mean)
    w_posteriror_cov = np.identity(len(weights))*b
    weights = np.array(weights)
    first = True
    num_datum = 0

    while first or not np.allclose(w_posteriror_mean[:, 0], w_prior_mean[:, 0], atol=0.0001, rtol=0):
    # while not np.allclose(w_posteriror_mean[:,0], weights, atol=0.3, rtol=0):
        print (np.sum(np.absolute(w_posteriror_mean - w_prior_mean), axis=0))
        data_x = 2 * np.random.random_sample(1) - 1
        data_x = data_x[0]
        e = normal_dist(0, np.sqrt(a))
        data_y = polynomial_dist(weights, data_x, e)
        w_prior_mean = np.copy(w_posteriror_mean)
        
        if first:
            X = np.array([[data_x**i for i in range(len(weights))]])
            y = np.array([[data_y]])
            w_posteriror_cov += a * np.dot(np.transpose(X), X)
            w_posteriror_mean = a * np.dot(np.linalg.inv(w_posteriror_cov), np.transpose(X)) * y[0, 0]
        else:
            X = np.append(X, np.array([[data_x**i for i in range(len(weights))]]), axis=0)
            y = np.append(y, np.array([[data_y]]), axis=0)
            s = np.copy(w_posteriror_cov)
            m = np.copy(w_posteriror_mean)
            w_posteriror_cov = a * np.dot(np.transpose(X), X) + s
            w_posteriror_mean = np.dot(np.linalg.inv(w_posteriror_cov), a * np.dot(np.transpose(X), y) + np.dot(s, m))

        design_x = np.array([data_x**i for i in range(len(weights))])
        y_predict_mean = np.dot(design_x, w_posteriror_mean)[0]
        y_predict_var = 1 / a + np.dot(np.dot(design_x, np.linalg.inv(w_posteriror_cov)), np.transpose(design_x))

        if first:
            predict_results = np.array([[data_x, y_predict_mean]])
            first = False
        else:
            predict_results = np.append(predict_results, [[data_x, y_predict_mean]], axis=0)

        print_info(data_x, data_y, w_posteriror_mean, np.linalg.inv(w_posteriror_cov), y_predict_mean, y_predict_var)
        num_datum += 1
        if num_datum == 10:
            plot_image(mode="running", weights=weights, e=e, a=y_predict_var, predict_results=predict_results, w_posteriror_cov=w_posteriror_cov)
        elif num_datum == 50:
            plot_image(mode="running", weights=weights, e=e, a=y_predict_var, predict_results=predict_results, w_posteriror_cov=w_posteriror_cov)

    print ("finish loss:", np.sum(np.absolute(w_posteriror_mean - w_prior_mean), axis=0))
    plot_image(mode="done", weights=weights, e=e, a=a, predict_results=predict_results)
