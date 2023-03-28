from segmentation import *
from registration import *
from utils import *
import numpy as np


def initialization(pcd, width, height):
    """
    Initialize the point cloud for coarsely registration.

    Args:
        pcd (ndarray): point cloud of US or CT
        width (int): width of the sternum (unit: mm)
        height (int): height of the sternum (unit: mm)

    Returns:
        labels (ndarray): a numpy array in size (n, 1). The label of each point.
        axises3D (ndarray): a numpy array in size (3, 3). The rotation matrix.
        translation (ndarray): a numpy array in size (1, 3). The translation vector that translate
            the averaged point to (0, 0, 0).
    """
    # pca and map to xy-plane
    translation = -np.mean(pcd, axis=0)  # demean
    pcd_trans = pcd + translation
    pcd_pca, axises3D = pca_transform(pcd_trans)
    pcd_xy = pcd_pca[:, :2]

    degree = 0

    ################## 1. match ##################
    degree = matching(
        pcd_xy,
        width,
        height,
        pixel_size=5,
        degree_start=0,
        degree_end=180,
        degree_step=10,
        loop=True,
        # save_path="match_1.svg"
    )

    # exit()
    ################## 2. match ##################
    degree = matching(
        pcd_xy,
        width,
        height,
        pixel_size=3,
        degree_start=degree - 15,
        degree_end=degree + 15,
        degree_step=2,
        # save_path="match_2.svg"
    )

    ################## 3. match ##################
    # rotate before match, more accurate.
    pcd_xy, axises3D_temp = rotation2D(pcd_xy, -degree)
    axises3D = axises3D_temp @ axises3D
    degree = matching(
        pcd_xy,
        width,
        height,
        pixel_size=2,
        degree_start=-7,
        degree_end=+7,
        degree_step=1,
        # save_path="match_3.svg"
    )
    pcd_xy, axises3D_temp = rotation2D(pcd_xy, -degree)
    axises3D = axises3D_temp @ axises3D

    pixels = pixelization(pcd_xy, pixel_size=2, min_points=1)
    # vis_pixels(pixels)

    ################## Begin segmentation ##################
    pixels, left, right = segment_pixels(
        pixels=pixels,
        line_scope=20,
        cluster_scope=2,
    )
    # vis_pixels(pixels)

    ################## Begin classification ##################
    pixels, axises3D_temp = classification(pixels, left, right)
    axises3D = axises3D_temp @ axises3D
    vis_pixels(pixels)
    # ################## Map result to 3D ##################
    labels = segment_3D(pcd_xy, pixels, 2)

    return labels, axises3D, translation


def matching(
    pcd_xy,
    width,
    height,
    pixel_size,
    degree_start,
    degree_end,
    degree_step,
    loop=False,
    save_path=None,
):
    """
    Template match algorithm for sternum detection. It will create a set of squares rotated in
    different degrees with input width and height. The degree of the square matching the sternum
    will be returned.

    Args:
        pcd_xy (ndarray): a numpy array in size (n, 2). Point cloud in xy-plane.
        width (int): width of the sternum (unit: mm)
        height (int): height of the sternum (unit: mm)
        pixel_size (int): The size of each pixel when discretizing (unit: mm)
        degree_start (int): The lower bound of rotation degree
        degree_end (int): The upper bound of rotation degree
        degree_step (int): The step of rotation degree
        loop (bool, optional): If the degree is from 0 to 180, set it to true. Defaults to False.
        save_path (string, optional): Path for saving the visualization results. Defaults to None.

    Returns:
        degree (int): Degree of the square matching the sternum
    """

    pixels = pixelization(pcd_xy, pixel_size=pixel_size, min_points=1)

    squares = create_squares(
        width=width,
        height=height,
        pixel_size=pixel_size,
        degree_start=degree_start,
        degree_end=degree_end,
        degree_step=degree_step,
    )

    feature_maps_square = square_match(pixels, squares)
    averaged_features_square = average_features_3D(
        feature_maps_square, radius_degree=3, radius_xy=3, loop=loop
    )
    coords, values = local_maxima_3D(
        averaged_features_square, radius_degree=5, radius_xy=3
    )
    try:
        degree = degree_start + coords[0, 0] * degree_step
        # vis_square_match(pixels, squares[coords[0, 0]], coords[0, 1:], save_path=save_path)
        return degree
    except IndexError:
        print("Can't find local maxima")
        exit()


def cal_ct_graph(ct_pcd_init, ct_pcd_sternum):
    """
    Calculate the graph of the CT image.

    Args:
        ct_pcd_init (ndarray): a numpy array in size (n, 3).
        pcd_sternum (ndarray): a numpy array in size (n, 3). Point cloud of sternum.

    Returns:
        ct_graph (graph): grah for the ct pcd, with attribute "pos" showing the nodes position.
    """
    graph = build_graph(ct_pcd_sternum)
    som_ct = GraphSom(ct_pcd_init, graph, batch_size=500, epochs=5000, radius=5)
    som_ct.train_som(pretrain=3)

    return som_ct.graph


def cal_us_graph(us_pcd_init, graph_ct):
    """
    Calculate the graph of the CT image.

    Args:
        ct_pcd_init (ndarray): a numpy array in size (n, 3).
        graph_ct (graph): grah for the ct pcd, with attribute "pos" showing the nodes' position.

    Returns:
        us_graph (graph): grah for the us pcd, with attribute "pos" showing the nodes' position.
    """
    som_us = GraphSom(
        us_pcd_init, graph_ct.copy(), batch_size=130, epochs=3000, radius=5
    )
    som_us.train_som()

    return som_us.graph


def compute_Rt(graph_ct, graph_us, radius=3):
    """
    Compute the rotation matrix and translation vector between two nodes.
    Collect the neighbor nodes' position in the radius of the node, and compute the Rt between the corresponding nodes
    in the two graphs by considering rigid transformation of the collected positions.

    Args:
        graph_ct (graph): grah for the ct pcd, with attribute "pos" showing the nodes' position.
        graph_us (graph): grah for the us pcd, with attribute "pos" showing the nodes' position.
        radius (int, optional): Defaults to 3.

    Returns:
        graph_ct (graph): The ct graph with node attribute "R" and "t".
    """
    graph_ct = collect_pos(graph_ct, radius)
    graph_us = collect_pos(graph_us, radius)
    graph_ct = nodes_Rt(graph_ct, graph_us)

    return graph_ct
