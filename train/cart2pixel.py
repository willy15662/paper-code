import pickle
import pandas as pd
import json
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

plt.rcParams.update({'font.size': 22})
import numpy as np

def find_duplicate(zp):
    dup = {}
    for i in range(len(zp[0, :])):
        for j in range(i + 1, len(zp[0])):
            if int(zp[0, i]) == int(zp[0, j]) and int(zp[1, i]) == int(zp[1, j]):
                dup.setdefault(str(zp[0, i]) + "-" + str(zp[1, i]), {i}).add(j)
    sum = 0
    for ind in dup.keys():
        sum += (len(dup[ind]) - 1)
    return sum

def dataset_with_best_duplicates(X, y, zp):
    X = X.transpose()
    dup = {}
    for i in range(len(zp[0, :])):
        for j in range(i + 1, len(zp[0])):
            if int(zp[0, i]) == int(zp[0, j]) and int(zp[1, i]) == int(zp[1, j]):
                dup.setdefault(str(zp[0, i]) + "-" + str(zp[1, i]), {i}).add(j)

    toDelete = []
    for index in dup.keys():
        mi = []
        x_new = X[:, list(dup[index])]
        mi = mutual_info_classif(x_new, y)
        max_index = np.argmax(mi)
        dup[index].remove(list(dup[index])[max_index])
        toDelete.extend(list(dup[index]))
    X = np.delete(X, toDelete, axis=1)
    zp = np.delete(zp, toDelete, axis=1)
    return X.transpose(), zp, toDelete

def count_model_col(rotatedData, Q, r1, r2, params=None):
    tot = []
    for f in range(r1 - 1, r2):
        A = int(f * 2 / 3)
        B = f
        xp = np.round(1 + (A * (rotatedData[0, :] - np.min(rotatedData[0, :])) / (np.max(rotatedData[0, :]) - np.min(rotatedData[0, :]))))
        yp = np.round(1 + (-B) * (rotatedData[1, :] - np.max(rotatedData[1, :])) / (np.max(rotatedData[1, :]) - np.min(rotatedData[1, :])))
        zp = np.array([xp, yp])
        A = np.max(xp)
        B = np.max(yp)

        # find duplicates
        sum_duplicates = str(find_duplicate(zp))
        print("Collisioni: " + sum_duplicates)
        tot.append([A, B, sum_duplicates])
        a = ConvPixel(Q["data"][:, 0], zp[0], zp[1], A, B)

        if params is not None:
            plt.savefig(params["dir"] + str(A) + "x" + str(B) + '.png')
        else:
            plt.savefig(str(A) + '.png')
        # plt.show()
    if params is not None:
        pd.DataFrame(tot, columns=["indexA", "indexB", "collisions"]).to_excel(params["dir"] + "Collisionmezzi.xlsx", index=False)
    else:
        pd.DataFrame(tot, columns=["index", "collisions"]).to_excel("Collision.xlsx", index=False)

def Cart2Pixel(Q=None, A=None, B=None, params=None):
    # TODO controls on input
    if A is not None:
        A = A - 1
    if B is not None:
        B = B - 1
    # to dataframe
    feat_cols = ["col-" + str(i + 1) for i in range(Q["data"].shape[1])]
    df = pd.DataFrame(Q["data"], columns=feat_cols)

    tsne = TSNE(n_components=2, method="exact")
    Y = tsne.fit_transform(df)

    x = Y[:, 0]
    y = Y[:, 1]
    n, n_sample = Q["data"].shape
    plt.scatter(x, y)
    bbox = minimum_bounding_rectangle(Y)
    plt.fill(bbox[:, 0], bbox[:, 1], alpha=0.2)
    # rotation
    grad = (bbox[1, 1] - bbox[0, 1]) / (bbox[1, 0] - bbox[0, 0])
    theta = np.arctan(grad)
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    bboxMatrix = np.array(bbox)
    zrect = (R.dot(bboxMatrix.T)).T
    plt.fill(zrect[:, 0], zrect[:, 1], alpha=0.2)

    coord = np.array([x, y])
    rotatedData = R.dot(coord)

    plt.scatter(rotatedData[0, :], rotatedData[1, :])
    plt.axis('square')
    plt.show(block=False)

    # find duplicate
    for i in range(len(rotatedData[0, :])):
        for j in range(i + 1, len(rotatedData[0])):
            if rotatedData[0, i] == rotatedData[0, j] and rotatedData[1, i] == rotatedData[1, j]:
                print("duplicate:" + str(i) + " " + str(j))

    xp = np.round(1 + (A * (rotatedData[0, :] - np.min(rotatedData[0, :])) / (np.max(rotatedData[0, :]) - np.min(rotatedData[0, :]))))
    yp = np.round(1 + (-B) * (rotatedData[1, :] - np.max(rotatedData[1, :])) / (np.max(rotatedData[1, :]) - np.min(rotatedData[1, :])))

    zp = np.array([xp, yp])
    A = np.max(xp)
    B = np.max(yp)

    # find duplicates
    print("Collisioni: " + str(find_duplicate(zp)))

    # Training set
    images = []
    name = "_" + str(int(A)) + 'x' + str(int(B))
  
    Q["data"], zp, toDelete = dataset_with_best_duplicates(Q["data"], Q["y"], zp)
    name += "_MI"

    image_model = {"xp": zp[0].tolist(), "yp": zp[1].tolist(), "A": A, "B": B, "toDelete": toDelete}
    with open(params["dir"] + "model" + name + ".json", "w") as f:
        json.dump(image_model, f)

    images = [ConvPixel(Q["data"][:, i], zp[0], zp[1], A, B, index=i) for i in range(n_sample)]

    with open(params["dir"] + "train" + name + ".pickle", 'wb') as f:
        pickle.dump(images, f)

    return images, image_model, toDelete

def ConvPixel(FVec, xp, yp, A, B, base=1, index=0):
    print(f"ConvPixel called with index={index}, A={A}, B={B}")

    n = len(FVec)
    M = np.ones([int(A), int(B)]) * base
    for j in range(n):
        x_idx = int(xp[j]) - 1
        y_idx = int(yp[j]) - 1
        if 0 <= x_idx < A and 0 <= y_idx < B:
            M[x_idx, y_idx] = FVec[j]
        else:
            print(f"Warning: Skipping out of bounds index at j={j}, xp={xp[j]}, yp={yp[j]}")
    
    zp = np.array([xp, yp])

    dup = {}
    # find duplicate
    for i in range(len(zp[0, :])):
        for j in range(i + 1, len(zp[0])):
            if int(zp[0, i]) == int(zp[0, j]) and int(zp[1, i]) == int(zp[1, j]):
                dup.setdefault(str(zp[0, i]) + "-" + str(zp[1, i]), {i}).add(j)

    print(f"ConvPixel completed for index={index}")
    return M

def minimum_bounding_rectangle(points):
    pi2 = np.pi / 2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = hull_points[1:] - hull_points[:-1]
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles - pi2),
        np.cos(angles + pi2),
        np.cos(angles)
    ]).T

    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval
