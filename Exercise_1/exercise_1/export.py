"""Export to disk"""
import numpy as np


def export_mesh_to_obj(path, vertices, faces):
    """
    exports mesh as OBJ
    :param path: output path for the OBJ file
    :param vertices: Nx3 vertices
    :param faces: Mx3 faces
    :return: None
    """

    # write vertices starting with "v "
    # write faces starting with "f "

    # ###############
    # TODO: Implement
    with open(path, 'w') as f:
        f.write("# OBJ file\n")
        for v in vertices:
            f.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))
        for face in faces:

            f.write("f %d %d %d\n" % (face[0]+1, face[1]+1, face[2]+1))

    # ###############


def export_pointcloud_to_obj(path, pointcloud):
    """
    export pointcloud as OBJ
    :param path: output path for the OBJ file
    :param pointcloud: Nx3 points
    :return: None
    """

    # ###############
    # TODO: Implement
    with open(path, 'w') as f:
        f.write("# OBJ file\n")
        for v in pointcloud:
            f.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))
    # ###############
