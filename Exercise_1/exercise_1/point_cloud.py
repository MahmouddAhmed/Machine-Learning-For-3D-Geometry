"""Triangle Meshes to Point Clouds"""
import numpy as np
import random
import math


def sample_point_cloud(vertices, faces, n_points):
    """
    Sample n_points uniformly from the mesh represented by vertices and faces
    :param vertices: Nx3 numpy array of mesh vertices
    :param faces: Mx3 numpy array of mesh faces
    :param n_points: number of points to be sampled
    :return: sampled points, a numpy array of shape (n_points, 3)
    """

    # ###############
    # TODO: Implement
    res = []
    areas = np.asarray([0.5*np.linalg.norm(np.cross(vertices[f][1]-vertices[f]
                                                    [0], vertices[f][2]-vertices[f][0]))for f in faces])
    areas = areas/np.sum(areas)
    total = areas.shape[0]

    for i in np.random.choice(range(total), size=n_points, p=areas):
        v1 = vertices[faces[i]][0]
        v2 = vertices[faces[i]][1]
        v3 = vertices[faces[i]][2]

        r1 = random.random()
        r2 = random.random()
        u = 1-math.sqrt(r1)
        v = math.sqrt(r1)*(1-r2)
        w = math.sqrt(r1)*r2

        v1 = v1*u
        v2 = v2*v
        v3 = v3*w
        res += [v1+v2 + v3]

    return np.array(res)

    # for f in faces:
    #     [v1, v2, v3] = vertices[f]
    #     print(v1)

    # areas = [0.5*np.linalg.norm(np.cross(v2-v1, v3-v1))
    #          for v1, v2, v3 in [f for f in vertices]]
    # p = areas/areas.sum()
    # print(areas.shape)
    # print(vertices.shape)
    # ###############
