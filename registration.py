import numpy as np
import networkx as nx


def build_graph(pcd_sternum):
    """
    build the directed graph and initailize the nodes location based on sternum position

    Args:
        pcd_sternum (ndarray): a numpy array in size (n, 3). Point cloud of sternum.

    Returns:
        graph (graph): directed graph with nodes attribute "pos" showing their initial position.
    """

    # build a matrix to represent the graph
    matrix = np.zeros((18, 21))
    for j in range(3, 18):
        for i in range(3):
            matrix[i, j] = 1

    for j in range(8, 13):
        for i in range(3, 6):
            matrix[i, j] = 1

    for j in range(2, 19):
        for i in range(6, 9):
            matrix[i, j] = 1

    for j in range(8, 13):
        for i in range(9, 11):
            matrix[i, j] = 1

    for j in range(1, 20):
        for i in range(11, 14):
            matrix[i, j] = 1

    for j in range(8, 13):
        for i in range(14, 15):
            matrix[i, j] = 1

    for j in range(0, 21):
        for i in range(15, 18):
            matrix[i, j] = 1

    x_min = np.min(pcd_sternum[:, 0])
    y_min = np.min(pcd_sternum[:, 1])
    x_max = np.max(pcd_sternum[:, 0])
    y_max = np.max(pcd_sternum[:, 1])
    pixel_size_x = (x_max - x_min) / 18
    pixel_size_y = (y_max - y_min) / 5

    def cal_pos(i, j):
        return np.array(
            [
                x_min + i * pixel_size_x,
                y_min + (j - matrix.shape[1] // 2 + 2.5) * pixel_size_y,
                0,
            ]
        )

    graph = nx.Graph()
    m, n = matrix.shape

    # convert the matrix to graph
    for i in range(m):
        for j in range(8, 12):
            if matrix[i, j]:
                graph.add_node((i, j), pos=cal_pos(i, j))
            if matrix[i, j + 1]:
                graph.add_node((i, j + 1), pos=cal_pos(i, j + 1))
                graph.add_edge((i, j), (i, j + 1))
            try:
                if matrix[i + 1, j]:
                    graph.add_node((i + 1, j), pos=cal_pos(i + 1, j))
                    graph.add_edge((i, j), (i + 1, j))
            except:
                pass

    for i in range(m):
        for j in range(8):
            if matrix[i, j] and matrix[i, j + 1]:
                graph.add_node((i, j), pos=cal_pos(i, j))
                graph.add_node((i, j + 1), pos=cal_pos(i, j + 1))
                graph.add_edge((i, j + 1), (i, j))

    for i in range(m):
        for j in range(12, n - 1):
            if matrix[i, j] and matrix[i, j + 1]:
                graph.add_node((i, j), pos=cal_pos(i, j))
                graph.add_node((i, j + 1), pos=cal_pos(i, j + 1))
                graph.add_edge((i, j), (i, j + 1))

    graph = graph.to_directed()

    for i in range(m - 1):
        for j in range(n):
            if matrix[i, j] and matrix[i + 1, j]:
                graph.add_node((i, j), pos=cal_pos(i, j))
                graph.add_node((i + 1, j), pos=cal_pos(i + 1, j))
                graph.add_edge((i + 1, j), (i, j))
                if i < 11:
                    graph.add_edge((i, j), (i + 1, j))

    graph = nx.convert_node_labels_to_integers(graph)

    for i in range(62, 87, 5):
        graph.add_edge(i, i + 5)

    graph.add_edge(87, 89)

    return graph


class GraphSom:
    """
    Graph based Self-Organizing Map
    """

    def __init__(
        self, points, graph, batch_size=500, epochs=10000, learning_rate=0.1, radius=5
    ):
        """

        Args:
            points (ndarray): the point cloud
            graph (graph): initialized graph with nodes attribute "pos" showing their initial position.
            batch_size (int, optional): Defaults to 500.
            epochs (int, optional): Defaults to 10000.
            learning_rate (float, optional): Defaults to 0.1.
            radius (int, optional): The nodes within this radius will will be update as well. Defaults to 5.
        """
        self.points = points
        self.graph = graph
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.radius = radius
        self.h = self.__cal_h()
        self.batch_size = batch_size
        self.neighbors = self.__find_neighbors(self.radius)

    def __update_lr(self, epoch):
        self.learning_rate = self.learning_rate * (1 - epoch / self.epochs)

    def __update_radius(self, epoch):
        new_radius = int((self.radius - 1) * (1 - epoch / self.epochs)) + 1
        if new_radius < self.radius:
            self.neighbors = self.__find_neighbors(new_radius)
            self.radius = new_radius
            self.h = self.__cal_h()

    def __cal_h(self):
        h = np.zeros((self.radius + 1, 3))
        for i in range(self.radius + 1):
            h[i] = np.exp(-(i**2) / (2 * self.radius**2))
        return h

    def __find_neighbors(self, radius):
        neighbors = {}
        for node in self.graph.nodes:
            neighbors[node] = nx.single_source_shortest_path_length(
                self.graph, node, cutoff=radius
            )
        return neighbors

    def __update_pos(self, batch, learning_rate):
        pos = self.get_pos()
        # calculate the distance between each point and each node
        dist = np.linalg.norm(pos[:, None] - batch[None], axis=2)
        # find the nearest node for each point
        idx = np.argmin(dist, axis=0)
        # update the position of each node
        for i, point in enumerate(batch):
            neighbors = self.neighbors[idx[i]]
            node = list(neighbors.keys())
            r = list(neighbors.values())
            pos[node] += self.h[r] * learning_rate * (point - pos[node])
        nx.set_node_attributes(self.graph, dict(zip(self.graph.nodes, pos)), "pos")

    def train_som(self, pretrain=0):
        for epoch in range(pretrain):
            self.__update_lr(epoch)
            self.__update_radius(epoch)
            batch_idx = np.random.randint(len(self.points), size=self.batch_size)
            batch = self.points[batch_idx]
            self.__update_pos(batch, 0.2 * self.learning_rate)
        # convert the graph to undirected after pretrain
        self.graph = self.graph.to_undirected()
        for epoch in range(self.epochs):
            self.__update_lr(epoch)
            self.__update_radius(epoch)
            batch_idx = np.random.randint(len(self.points), size=self.batch_size)
            batch = self.points[batch_idx]
            self.__update_pos(batch, self.learning_rate)

    def get_pos(self):
        node_pos = nx.get_node_attributes(self.graph, "pos")
        pos = np.array([node_pos[node] for node in self.graph.nodes])
        return pos


def collect_pos(graph, radius):
    """
    Collect positions of the nodes within the radius in the graph.

    Args:
        graph (nx.Graph): The graph with node attribute "pos".
        radius (int): radius.

    Returns:
        nx.Graph: The graph with node attribute "neghbor_pos".
    """

    node_pos = nx.get_node_attributes(graph, "pos")
    nodes = graph.nodes()
    for n in nodes:
        neighbors = nx.single_source_shortest_path_length(graph, n, cutoff=radius)
        neighbors_pos = []
        for node, _ in neighbors.items():
            neighbors_pos.append(node_pos[node])
        graph.nodes[n]["neighbor_pos"] = np.array(neighbors_pos)

    return graph


def rigid_transform_3D(A, B):
    """
    B = (R @ A.T).T + t
    A is the CT (source), B is the US (target)

    Args:
        A (ndarray): in shape (n, 3), represents n points in 3D space.
        B (ndarray): in shape (n, 3), represents n points in 3D space.

    Returns:
        R (ndarray): in shape (3, 3), the rotation matrix.
        t (ndarray): in shape (3, 1), the translation vector.
    """
    assert len(A) == len(B)
    N = A.shape[0]
    mu_A = np.mean(A, axis=0)
    mu_B = np.mean(B, axis=0)

    AA = A - np.tile(mu_A, (N, 1))
    BB = B - np.tile(mu_B, (N, 1))
    H = np.transpose(AA) @ BB

    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        # Reflection detected
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ mu_A.T + mu_B.T

    return R, t


def nodes_Rt(graph_A, graph_B):
    """
    Compute the rigid transformation for each nodes in graph_A to graph_B.
    The graph should contain the node attribute "neighbor_pos".

    Args:
        graph_A (nx.Graph): A is the grah of CT point cloud
        graph_B (nx.Graph): B is the grah of US point cloud

    Returns:
        nx.Graph: The graph with node attribute "R" and "t".
    """

    node_pos_A = nx.get_node_attributes(graph_A, "neighbor_pos")
    node_pos_B = nx.get_node_attributes(graph_B, "neighbor_pos")
    nodes = graph_A.nodes()
    for n in nodes:
        R, t = rigid_transform_3D(node_pos_A[n], node_pos_B[n])
        graph_A.nodes[n]["R"] = R
        graph_A.nodes[n]["t"] = t

    return graph_A


def transform_nodes(graph):
    """
    Transform the nodes in the graph. The graph should contain the node attribute "R" and "t".
    The transformed nodes will be stored in the node attribute "new_pos".

    Args:
        graph (nx.Graph): a graph with node attribute "R" and "t". Put the CT graph here.
    """

    node_pos = nx.get_node_attributes(graph, "pos")
    nodes = graph.nodes()
    for n in nodes:
        R = graph.nodes[n]["R"]
        t = graph.nodes[n]["t"]
        node_pos[n] = (R @ node_pos[n].T).T + t
    nx.set_node_attributes(graph, node_pos, "new_pos")

    return graph


def tramsform_pcd(pcd, graph, k):
    """
    Transform the pcd using the graph. The graph should contain the node attribute "pos", "R", "t".

    Args:
        pcd (np.ndarray): The pcd to be transformed.
        graph (nx.Graph): The graph with node attribute "pos", "R", "t".
        k (int): The number of nearest nodes to be used.

    Returns:
        pcd_trans (ndarray): The transformed pcd.
    """

    node_pos = nx.get_node_attributes(graph, "pos")
    pos = [node_pos[n] for n in graph.nodes()]
    node_R = nx.get_node_attributes(graph, "R")
    node_t = nx.get_node_attributes(graph, "t")

    pcd_trans = []
    for p in pcd:
        dist = np.linalg.norm(pos - p, axis=1)
        index = np.argsort(dist)
        index = index[:k]
        dist = dist[index]
        weights = dist / np.sum(dist)

        p_list = []
        for i in range(k):
            p_list.append((node_R[index[i]] @ p.T).T + node_t[index[i]])

        p = np.average(p_list, axis=0, weights=weights)

        pcd_trans.append(p)

    return np.array(pcd_trans)


def transform_points(points_source, pcd_source, pcd_transformed, r=20):
    """
    transform the points in points_source space to the points in pcd_transformed space.
    The points in the pcd_source and pcd_transformed should be in the same order, i.e., when 
    transforming the i-th point in pcd_source space to pcd_transformed space, the transformed
    point should be the i-th point in pcd_transformed space.
    The points in the ball with radius r will be used to calculate the transformation matrix.

    Args:
        points_source (ndarray): in shape (n, 3), represents n points in 3D space.
        pcd_source (ndarray): the sorce point cloud.
        pcd_target (ndarray): the transformed point cloud.
        r (int, optional): radius. Defaults to 20.

    Returns:
        points_target (ndarray):  in shape (n, 3)
    """
    points_target = []
    for point in points_source:
        dist = np.linalg.norm(pcd_source - point, axis=1)
        pcd_source_r = pcd_source[dist < r]
        pcd_target_r = pcd_transformed[dist < r]
        # calculate the transformation matrix that transforms pcd_source_r to pcd_target_r
        R, t = rigid_transform_3D(pcd_source_r, pcd_target_r)
        point = (R @ point + t).reshape(-1)
        points_target.append(point)

    return np.array(points_target)
