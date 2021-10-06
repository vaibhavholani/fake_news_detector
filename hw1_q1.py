import numpy as np
from typing import List
import matplotlib.pyplot as plt

def get_sample_points(num: int, d: int):
    return np.random.rand(num, d)

def find_all_distances(sample_points: List[float], d: int):

    all_distances = []
    iter_range = len(sample_points)
    for i in range(iter_range):
        for j in range(i+1, iter_range):
            all_distances.append(euclidan_distance(sample_points[i], sample_points[j], 2**d))

    if len(all_distances) == 0:
        return [0,0,d]
    else:
        return [np.mean(all_distances), np.std(all_distances), 2**d]

def euclidan_distance(x: List, y: List, d: int):
    total = 0
    for i in range(d):
        total += (x[i]-y[i])**2
    return total


def plot_stats(all_stats):

    y1 = [y[0] for y in all_stats]
    y2 = [y[1] for y in all_stats]
    d = [y[2] for y in all_stats]

    plt.plot(d, y1, label = "Mean of Distances")
    plt.plot(d, y2, label = "Std. Dev. of Distances")

    plt.xlabel('Dimension')
    plt.ylabel("Mean and Std. Dev.")

    plt.legend()
    plt.show()


def main():

    all_stats = []
    for i in range(11):
        points = get_sample_points(100, 2**i)
        sample_stats = find_all_distances(points, i)
        all_stats.append(sample_stats)
    plot_stats(all_stats)

if __name__ == '__main__':
    main()