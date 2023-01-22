"""Creating an SDF grid"""
import numpy as np


def sdf_grid(sdf_function, resolution):
    """
    Create an occupancy grid at the specified resolution given the implicit representation.
    :param sdf_function: A function that takes in a point (x, y, z) and returns the sdf at the given point.
    Points may be provides as vectors, i.e. x, y, z can be scalars or 1D numpy arrays, such that (x[0], y[0], z[0])
    is the first point, (x[1], y[1], z[1]) is the second point, and so on
    :param resolution: Resolution of the occupancy grid
    :return: An SDF grid of specified resolution (i.e. an array of dim (resolution, resolution, resolution) with positive values outside the shape and negative values inside.
    """

    # ###############
    # TODO: Implement
    world_cordinates = np.linspace(-0.5, 0.5, resolution)
    xv, yv, zv = np.meshgrid(
        world_cordinates, world_cordinates, world_cordinates)
    vox_coords = np.stack([xv.flatten(), yv.flatten(), zv.flatten()])
    grid = sdf_function(vox_coords[1], vox_coords[0], vox_coords[2])
    grid = grid.reshape((resolution, resolution, resolution))
    # world_cordinates = np.linspace(-0.5, 0.5, resolution)
    # grid = np.fromfunction(lambda i, j, k: sdf_function(
    #     world_cordinates[i], world_cordinates[j], world_cordinates[k]), (resolution, resolution, resolution), dtype=int)
    return grid
    # ###############
