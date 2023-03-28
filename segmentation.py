import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from math import ceil
from scipy import ndimage as ndi
from skimage.segmentation import flood_fill


def pca_transform(pcd):
    """
    apply pca transform to the point cloud.

    Args:
        pcd (ndarray): The point cloud

    Returns:
        pcd_pca (ndarray): The transformed point cloud
        axises (ndarray): The main axis of the input point cloud
    """
    pca = PCA(n_components=3)
    pca.fit(pcd)
    axises = pca.components_  # type: ignore
    for i in range(3):
        if axises[i][i] < 0:
            axises[i] = -axises[i]
    pcd_pca = (axises @ pcd.T).T
    return pcd_pca, axises


def rotation2D(inputs, degree):
    """
    Rotate the 2D numpy array according to the input degree.

    Args:
        inputs (ndarray): A 2D numpy array in shape (n, 2)
        degree (int): degree for the rotation

    Returns:
        rotated (ndarray): rotated numpy array in shape (n, 2)
        axises3D (ndarray): the correspondent 3D rotation matrix in shape (3, 3)
    """
    rad = np.radians(degree)
    sin_rad = np.sin(rad)
    cos_rad = np.cos(rad)
    M = np.array([[cos_rad, -sin_rad], [sin_rad, cos_rad]])
    rotated = (M @ inputs.T).T
    axises3D = np.array([[cos_rad, -sin_rad, 0], [sin_rad, cos_rad, 0], [0, 0, 1]])
    return rotated, axises3D


def pixelization(pcd_xy, pixel_size, min_points=1):
    """
    Pixelize the point cloud in xy plane.

    Args:
        pcd_xy (ndarray): a numpy array in size (n, 2)
        pixel_size (int): The size of the pixels (unit: mm)
        min_points (int, optional):Mark the pixel as TRUE if at least min_points
            points in this pixels. Defaults to 1.

    Returns:
        pixels (ndarray): The pixelized point cloud as a boolean numpy array
    """

    lb_x = pcd_xy[:, 0].min()
    ub_x = pcd_xy[:, 0].max()

    lb_y = pcd_xy[:, 1].min()
    ub_y = pcd_xy[:, 1].max()

    pixels = np.array([])
    for x in np.arange(lb_x, ub_x, pixel_size):
        for y in np.arange(lb_y, ub_y, pixel_size):
            pixels = np.append(
                pixels,
                len(
                    pcd_xy[
                        (pcd_xy[:, 0] >= x)
                        & (pcd_xy[:, 0] < (x + pixel_size))
                        & (pcd_xy[:, 1] >= y)
                        & (pcd_xy[:, 1] < (y + pixel_size))
                    ]
                ),
            )

    pixels = pixels >= min_points
    return pixels.reshape(
        ceil(((ub_x - lb_x) / pixel_size)), ceil(((ub_y - lb_y) / pixel_size))
    )


def create_squares(width, height, pixel_size, degree_start, degree_end, degree_step):
    """
    Create a set of square templates.

    Args:
        width (int): The length of the shorter side of the square (unit: mm)
        height (int): The length of the longer side of the square (unit: mm)
        pixel_size (int): The size of the pixels
        degree_start (int): Rotation degree for the first square
        degree_end (int): Rotation degree for the last square (may not included)
        degree_step (int): The step of rotation degree

    Returns:
        squares (ndarray): In shape of (#squres, template_size[0], template_size[1])
    """

    # create initial template
    template_coordinates = np.array(
        [
            [i, j]
            for i in np.arange(0, width, pixel_size / 10)
            for j in np.arange(0, height, pixel_size / 10)
        ]
    )

    template_coordinates[:, 0] -= width / 2
    template_coordinates[:, 1] -= height / 2

    # calculate rotation matrices
    rads = np.radians(np.arange(degree_start, degree_end, degree_step))
    sin_rads = np.sin(rads)
    cos_rads = np.cos(rads)
    M = np.array(
        [
            [[cos_rads[i], -sin_rads[i]], [sin_rads[i], cos_rads[i]]]
            for i in range(len(rads))
        ]
    )

    # apply rotation to the tamplate
    rotated_coordinates = (M @ template_coordinates.T).transpose((0, 2, 1))
    # pixelization
    # tranlate the templates, make the min of x and y be 0
    rotated_coordinates[:, :, 0] -= rotated_coordinates[:, :, 0].min()
    rotated_coordinates[:, :, 1] -= rotated_coordinates[:, :, 1].min()
    rotated_coordinates = (rotated_coordinates / pixel_size).astype(np.int32)

    squares = np.zeros(
        (
            rotated_coordinates.shape[0],
            np.max(rotated_coordinates[:, :, 0]) + 1,
            np.max(rotated_coordinates[:, :, 1]) + 1,
        ),
        dtype=bool,
    )

    # Make the size of template odd
    if squares.shape[1] % 2 != 1:
        squares = np.insert(squares, 0, values=0, axis=1)

    if squares.shape[2] % 2 != 1:
        squares = np.insert(squares, 0, values=0, axis=2)

    for i in range(rotated_coordinates.shape[0]):
        mask = np.unique(rotated_coordinates[i], axis=0)
        squares[i][mask[:, 0], mask[:, 1]] = True

    return squares


def square_match(pixels, squares):
    """
    Match the squares by counting how many pixels in a region of pixels match the
    square (logical and).

    Args:
        pixels (ndarray): In shape (m, n). The pixelized point cloud.
        templates (ndarray): In shape (#squres, template_size[0], template_size[1]).
            A set of square templates.

    Returns:
        feature_maps (ndarray): In shape (#squres, m, n). The values of index (i, j, k) means
            the number of matched pixels when match the i-th square with the region of pixels
            centering at (j, k).
    """

    m = squares.shape[1] // 2
    n = squares.shape[2] // 2
    paded_pixels = np.expand_dims(
        np.pad(pixels, ((m, m), (n, n))),  # type: ignore
        axis=0,
    )
    feature_maps = np.zeros((squares.shape[0], pixels.shape[0], pixels.shape[1]))
    # vis_square_match(pixels, squares[0], (0, pixels.shape[1]//2), save_path='match.svg')

    # begin convolution
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            crop = paded_pixels[:, i : i + squares.shape[1], j : j + squares.shape[2]]
            feature_maps[:, i, j] = (crop * squares).sum(axis=(1, 2))

    return feature_maps


def average_features_3D(feature_maps, radius_degree, radius_xy, loop=False):
    """
    For each element in the input feature maps, replace its value by averaging its and its
    neighbours' values within the radius (±radius_degree, ±radius_xy, ±radius_xy).

    Args:
        feature_maps (ndarray): Feature maps in shape (#squres, m, n)
        radius_degree (int): The average raidus for degrees
        radius_xy (int): The average radius for pixels
        loop (bool, optional): Set it to True if the first degree and the last degree is
            neighbored. E.g. 180 and 0. Defaults to False.

    Returns:
        averaged_features (ndarry): In shape (#squres, m, n). The values of (i, j, k) means
            the averaged number of matched pixel when matching the i-th square with the region of
            pixels centering at (j, k) calculated by averaging the region (i ± radius_degree,
            j ± radius_xy, k ± radius_xy)
    """

    feature_maps_padded = np.pad(
        feature_maps,
        (
            (radius_degree, radius_degree),
            (radius_xy, radius_xy),
            (radius_xy, radius_xy),
        ),
    )  # type: ignore

    # if the first degree and the last degree is neighbored. E.g. 180 and 0.
    if loop:
        for i in range(radius_degree):
            feature_maps_padded[i] = feature_maps_padded[i - (2 * radius_degree)]
            feature_maps_padded[-i - 1] = feature_maps_padded[
                -i - 1 + (2 * radius_degree)
            ]
    else:
        for i in range(radius_degree):
            feature_maps_padded[i] = feature_maps_padded[radius_degree]
            feature_maps_padded[-i - 1] = feature_maps_padded[-1 - radius_degree]

    averaged_features = np.zeros(feature_maps.shape)
    range_degree = 1 + 2 * radius_degree
    range_xy = 1 + 2 * radius_xy
    for i in range(averaged_features.shape[0]):
        for j in range(averaged_features.shape[1]):
            for k in range(averaged_features.shape[2]):
                averaged_features[i, j, k] = np.average(
                    feature_maps_padded[
                        i : i + range_degree,
                        j : j + range_xy,
                        k : k + range_xy,
                    ]
                )

    return averaged_features


def average_features_1D(feature_maps, radius):
    """Average a 1D ndarray.

    Args:
        feature_maps (ndarray): in shape (n, )
        radius (int): averge radius

    Returns:
        averaged_features (ndarry): In shape (n, ). The values of of index i means the averaged
            value of region (i - radius, i + radius)
    """

    averaged_features = np.zeros(feature_maps.shape)
    for i in range(len(feature_maps)):
        averaged_features[i] = np.average(feature_maps[max(0, i - radius) : i + radius])

    return averaged_features


def local_maxima_3D(data, radius_degree, radius_xy, loop=False):
    """
    Detects local maxima in a 3D array. Sort the local maxima in descending order.

    Args:
        data (ndarray): 3d ndarray
        scope (int, optional): How many points on each side to use for the comparison. Defaults to 1.

    Returns:
        coords (ndarray): coordinates of the local maxima
        values (ndarray): values of the local maxima
    """

    data = np.pad(data, ((radius_degree, radius_degree), (radius_xy, radius_xy), (radius_xy, radius_xy)))  # type: ignore
    # if the first degree and the last degree is neighbored. E.g. 180 and 0.
    if loop:
        for i in range(radius_degree):
            data[i] = data[i - (2 * radius_degree)]
            data[-i - 1] = data[-i - 1 + (2 * radius_degree)]

    size_xy = 1 + 2 * radius_xy
    size_degree = 1 + 2 * radius_degree
    footprint = np.ones((size_degree, size_xy, size_xy))
    footprint[radius_degree, radius_xy, radius_xy] = 0

    filtered = ndi.maximum_filter(data, footprint=footprint)
    filtered = filtered[
        radius_degree:-radius_degree, radius_xy:-radius_xy, radius_xy:-radius_xy
    ]
    data = data[
        radius_degree:-radius_degree, radius_xy:-radius_xy, radius_xy:-radius_xy
    ]
    mask_local_maxima = data > filtered
    coords = np.asarray(np.where(mask_local_maxima)).T
    values = data[mask_local_maxima]

    index_sorted = values.argsort()[::-1]
    coords = coords[index_sorted]
    values = values[index_sorted]

    return coords, values


def segment_pixels(pixels, line_scope, cluster_scope):
    """
    Segment at pixel's level. Find the bound of sternum using line match. Using K-mean to
    segment the cartilages near to the boundry and the growing the region using flood fill.

    Args:
        pixels (ndarray): the pixelized pcd
        line_scope (int): the scope for finding the sternum bound from the center
        cluster_scope (int): the scope for clustering the cartilages from the sternum bound

    Returns:
        pixels (ndarray): the pixelized pcd with the segmented sternum and cartilages. The value n of
            element (i, j) means the points in this pixel belong to categories n. The value 1 means
            the points belong to the sternum. The value lager than 1 means the points belong to
            the cartilages.
        left (int): the left bound of the sternum
        right (int): the right bound of the sternum

    """

    # segment sternum
    center = pixels.shape[0] // 2
    feature_maps_sternum = np.sum(pixels, axis=1)
    feature_maps_sternum = average_features_1D(feature_maps_sternum, 1)

    feature_maps_l = feature_maps_sternum[(center - line_scope) : center]
    feature_maps_r = feature_maps_sternum[center : (center + line_scope)]

    for i in range(len(feature_maps_l) - 1, 0, -1):
        feature_maps_l[i] = feature_maps_l[i] - feature_maps_l[i - 1]
    feature_maps_l[0] = 0

    for i in range(len(feature_maps_r) - 1):
        feature_maps_r[i] = feature_maps_r[i] - feature_maps_r[i + 1]
    feature_maps_r[-1] = 0

    left = center - (len(feature_maps_l) - np.argmax(feature_maps_l)) - 2
    right = center + np.argmax(feature_maps_r) + 1

    # clustering pixels close to the boundry
    pixels = pixels.astype(np.int8)
    # up / left
    X = np.c_[np.where(pixels[max(0, left - cluster_scope + 1) : left + 1, :] != 0)]
    X[:, 0] += max(0, left - cluster_scope + 1)
    kmeans = KMeans(n_clusters=4).fit(X)
    pixels[X[:, 0], X[:, 1]] = kmeans.labels_ + 2  # type: ignore
    # down / right
    X = np.c_[np.where(pixels[right : right + cluster_scope, :] != 0)]
    kmeans = KMeans(n_clusters=4).fit(X)
    X[:, 0] += right
    pixels[X[:, 0], X[:, 1]] = kmeans.labels_ + 6  # type: ignore
    # vis_pixels(pixels)

    # keep the labels consistant
    orders_up = []
    for label in pixels[left, :]:
        if label > 1 and label not in orders_up:
            orders_up.append(label)
    orders_down = []
    for label in pixels[right, :]:
        if label > 1 and label not in orders_down:
            orders_down.append(label)
    for i in range(4):
        pixels[pixels == orders_down[i]] = orders_up[i]

    # beging region growing to segment cartilages
    # up / left
    for i in range(max(0, left - cluster_scope), -1, -1):
        no_class = np.where(pixels[i, :] == 1)
        for ind in no_class[0][::-1]:
            if pixels[i + 1, ind] > 1:
                pixels = flood_fill(pixels, (i, ind), pixels[i + 1, ind])

    # down / right
    for i in range(right + cluster_scope, pixels.shape[0]):
        no_class = np.where(pixels[i, :] == 1)
        for ind in no_class[0][::-1]:
            if pixels[i - 1, ind] > 1:
                pixels = flood_fill(pixels, (i, ind), pixels[i - 1, ind])

    return pixels, left, right


def classification(pixels, left, right, buffer_scope=5):
    """
    Classify the cartilages based on the curvature of the midline. Pixels for the 2-nd cartilage is
    set to value 2, piexls for the 3-rd cartilage is set to value 3, and so on.

    Args:
        pixels (ndarray): The pixelized pcd with the segmented sternum and cartilages.
        left (int): the left bound of the sternum
        right (int): the right bound of the sternum
        buffer_scope (int, optional): Weight for midline change near the boundaries, as the segmentation
            may not accurate. Defaults to 5.

    Returns:
        pixels (ndarray): The pixelized pcd with the segmented sternum and cartilages. The 2nd cartilage
            is set to value 2, the 3rd cartilage is set to value 3, and so on.
        axises3D (ndarray): the 3D transformation matrix, that let the 2nd cartilage be the upper one.
    """

    # current order of the upper boundary
    orders = []
    for label in pixels[left, :]:
        if label > 1 and label not in orders:
            orders.append(label)

    # the midline of the first cartilage and last cartilage
    midline_first = []
    midline_last = []
    for i in range(pixels.shape[0]):
        ind = np.where(pixels[i, :] == orders[0])[0]
        if ind.size > 0:
            ind = np.sort(ind)
            midline_first.append((ind[0] + ind[-1]) / 2)
        ind = np.where(pixels[i, :] == orders[-1])[0]
        if ind.size > 0:
            ind = np.sort(ind)
            midline_last.append((ind[0] + ind[-1]) / 2)

    # calculate the midline change, if the pixel is very close to the boundary, the change is smaller by 0.5, as the segmentation may not accurate
    feature_first = 0
    feature_last = 0
    for i in range(len(midline_first) - 1):
        if i < buffer_scope:
            feature_first += 0.5 * abs(midline_first[i] - midline_first[i + 1])
        else:
            feature_first += abs(midline_first[i] - midline_first[i + 1])
    for i in range(len(midline_last) - 1):
        if i < buffer_scope:
            feature_last += 0.5 * abs(midline_last[i] - midline_last[i + 1])
        else:
            feature_last += abs(midline_last[i] - midline_last[i + 1])

    if feature_first > feature_last:
        pixels[pixels == (orders[3])] = 6
        pixels[pixels == (orders[2])] = 7
        pixels[pixels == (orders[1])] = 8
        pixels[pixels == (orders[0])] = 9
        axises3D = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    else:
        pixels[pixels == (orders[0])] = 6
        pixels[pixels == (orders[1])] = 7
        pixels[pixels == (orders[2])] = 8
        pixels[pixels == (orders[3])] = 9
        axises3D = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    pixels[pixels > 1] -= 4

    # vis_pixels(pixels)

    return pixels, axises3D


def segment_3D(pcd_xy, pixels, pixel_size):
    """
    Map the pixel level segmentation to 3D points

    Args:
        pcd_xy (ndarray): a numpy array in size (n, 2)
        pixels (_type_): The pixelized pcd after classification
        pixel_size (int): The size of the pixels (unit: mm)

    Returns:
        labels: Labels for the point clouds, following the same order as the pcd array.
            the pcd array is in shape (n, 3) <=> the label is in shape (n, ).
    """
    pcd_xy_temp = np.copy(pcd_xy)
    pcd_xy_temp[:, 0] -= pcd_xy_temp[:, 0].min()
    pcd_xy_temp[:, 1] -= pcd_xy_temp[:, 1].min()

    pcd_xy_temp //= pixel_size
    pcd_xy_temp = pcd_xy_temp.astype(np.int8)

    labels = np.zeros(len(pcd_xy_temp), dtype=np.int8)
    for ind, (i, j) in enumerate(pcd_xy_temp):
        labels[ind] = pixels[i, j]

    return labels
