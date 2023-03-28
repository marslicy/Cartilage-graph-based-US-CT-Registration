import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def vis_pcd(points, point_size, title=""):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for i, pcd in enumerate(points):
        ax.scatter(
            pcd[:, 0], pcd[:, 1], pcd[:, 2], s=point_size[i], linewidths=0, alpha=1, marker=".", label=i  # type: ignore
        )
    plt.title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")  # type: ignore
    plt.show()


def vis_pixels(pixels, title="", vertexes=None, save_path=None):
    plt.figure()
    if vertexes is not None:
        pixels = pixels.astype(np.int32)
        for v in vertexes:
            pixels[v] = 5
    # pixels[np.where(pixels == 1)] = 6
    # colors = ['#440154', '#30678D', '#35B778', '#FDE724', '#DC143C']
    # cmap = mpl.colors.ListedColormap(colors)
    plt.imshow(pixels)
    plt.xlabel("y")
    plt.ylabel("x")
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, format="svg")
    plt.show()


def vis_scatter(pcd_2d, point_size):
    for i, pcd in enumerate(pcd_2d):
        x = pcd[:, 0]
        y = pcd[:, 1]
        plt.scatter(x, y, s=point_size[i], linewidths=0, alpha=1, marker=".", label=i)

    plt.show()


def vis_square_match(pixels, template, coords, save_path=None):
    # vis_square_match(us_pixels, squares[coords[0, 0]], coords[0, 1:])
    m = template.shape[0] // 2
    n = template.shape[1] // 2
    pixels = np.pad(pixels, ((m, m), (n, n)))  # type: ignore

    ind_x = coords[0]
    ind_y = coords[1]

    pixels = pixels.astype(np.int32)

    template = template.astype(np.int32)
    template += 2
    pixels[
        ind_x : ind_x + template.shape[0], ind_y : ind_y + template.shape[1]
    ] += template

    vis_pixels(pixels, save_path=save_path)


def plt_point_cloud(points):
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, linewidths=0, alpha=1, marker=".")  # type: ignore
    plt.title("Point Cloud")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")  # type: ignore
    plt.show()


def cal_hausdorff_distance(points_A, points_B):
    """
    Calculate the hausdorff distance between two point clouds.

    Args:
        points_A (np.ndarray): The first point cloud.
        points_B (np.ndarray): The second point cloud.

    Returns:
        float: The hausdorff distance.
    """
    points_A = np.array(points_A)
    points_B = np.array(points_B)
    points_A = points_A[np.random.choice(points_A.shape[0], 1600, replace=False)]
    points_B = points_B[np.random.choice(points_B.shape[0], 1600, replace=False)]
    dist = np.linalg.norm(points_A[:, None] - points_B, axis=2)
    dist_A = np.min(dist, axis=1)
    dist_A = np.sort(dist_A)

    dist_B = np.min(dist, axis=0)
    dist_B = np.sort(dist_B)

    compact = np.array([dist_A, dist_B])

    return np.max(compact, axis=0)


def vis_graph(graph):
    node_pos = nx.get_node_attributes(graph, "pos")
    for k, v in node_pos.items():
        node_pos[k] = np.array([-v[1], -v[0]])
    nx.draw(graph, node_pos, with_labels=graph.nodes, node_size=30, font_size=8)
    # plt.savefig("./graph.svg", dpi=300,format='svg')
    plt.show()
