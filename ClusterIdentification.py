# Clustering methods from Tereda's paper
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm


def kmeans(data, k=5, limit=500):
    """
    K-means clustering.
    ref: http://ixora.io/itp/learning_machines/clustering-and-numpy/

    :param data: cells * positions value=dff
    :param k: pick the first k cells to be the centers
    :param limit:
    :return:
    """
    centers = data[:k, :]

    for i in range(limit):
        classifications = np.argmin(((data[:, :, None] -
                                      centers.T[None, :, :]) ** 2).sum(axis=1),
                                    axis=1)

        new_centers = np.array([data[classifications == j, :].mean(axis=0)
                                for j in range(k)])
        if (new_centers == centers).all():
            break
        else:
            centers = new_centers

    return classifications, centers


def classify(k, pl_activity, plot=True):  # pl_activity:[pl_id, activity over positions]
    """
    Classify cells into several clusters according to their firing patterns.

    :param k: number of clusters
    :param pl_activity: [pl_id, activity over positions]
    :param plot: whether make plot or not
    :return:
    classifications: the cluster id of each cell
    centers: centers of each cluster
    """
    classifications, centers = kmeans(data=pl_activity, k=k)  # classifications: the cluster id of each cell
    centers_peak = np.zeros(k)
    for i in range(k):
        centers_peak[i] = np.argmax(centers[i, 1:-1])
    centers_rank = np.argsort(centers_peak)
    #     print(centers_rank)
    if plot:
        for m in range(k):
            plt.plot(k - m + centers[centers_rank[m], 1:-1])
            plt.fill_between(x=np.arange(1, 200, 1), y1=k - m, y2=k - m + centers[centers_rank[m], 1:-1])
        plt.xlabel('Position(cm)')
        plt.ylabel('Average Cluster')
        plt.show()
    return classifications, centers


def silhouette_value_point(point, data, classifications):
    """
    Computing silhouette value.

    :param point: the id of the cluster:[cluster, data of the cluster]
    :param data: [cluster, data of the cluster]
    :param classifications: [cluster id of each cluster]
    :return:
    s: computed silhouette value
    """
    cluster_num = np.max(classifications) + 1
    clusters = [[] for i in range(cluster_num)]
    for i in range(cluster_num):
        clusters[i]= np.where(classifications==i)[0]
    coh = 0
    for p in clusters[classifications[point]]:
        coh += norm(data[p]-data[point])
    coh /= len(clusters[classifications[point]])
    seps = []
    for k in range(cluster_num):
        if k != classifications[point]:
            sep = 0
            for p in clusters[k]:
                sep += norm(data[p]-data[point])
            sep /= len(clusters[k])
            seps.append(sep)
    sep = np.min(np.array(seps))
    s = (sep-coh) / max(sep, coh)
    return s


def silhouette_score(data, classifications):
    """
    Compute silhouette score.

    :param data: activities of all cells
    :param classifications: [cluster id of each cluster]
    :return: several silhouette score
    """
    s = []
    for p in range(data.shape[0]):
        s.append(silhouette_value_point(p, data, classifications))
    s = np.array(s)
    s_mean = np.mean(s)
    s = np.sort(s)
    s_25 = s[int(len(s)*0.25)]
    s_50 = s[int(len(s)*0.5)]  # median
    s_75 = s[int(len(s)*0.75)]
    return s_mean, s_25, s_50, s_75


def find_best_k(data):
    """
    Adding one additional cluster improve the median silhouette value the most was taken as the estimate of k

    :param data: activities of all cells
    :return: the best number of clusters
    """
    medians = np.zeros(data.shape[0])
    for k in range(2, data.shape[0]):
        classification = classify(k, data, plot=False)[0]
        medians[k] = silhouette_score(data, classification)[2]
    delta_medians = medians[1:]-medians[:-1]
    k = np.argmax(delta_medians)+2
    return k