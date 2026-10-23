import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

try:
    import gco
except ImportError:
    gco = None  # handle gco missing gracefully if needed

def show_matching_result(img1, img2, keypoints1, keypoints2):
    """
    Plot the images and their corresponding matching points.
    
    Args:
        img1 (np.ndarray): First input image.
        img2 (np.ndarray): Second input image.
        keypoints1 (np.ndarray): Array of matching keypoints from img1.
        keypoints2 (np.ndarray): Array of matching keypoints from img2.
    """
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(np.hstack((img1, img2)), cmap="gray")

    for kp1, kp2 in zip(keypoints1, keypoints2):
        plt.scatter(kp1[0], kp1[1], s=10, edgecolors='white', c="tab:blue")
        plt.scatter(kp2[0] + img1.shape[1], kp2[1], s=10, edgecolors='white', c="tab:blue")
        plt.plot([kp1[0], kp2[0] + img1.shape[1]], [kp1[1], kp2[1]])


def get_affine_transformation(
    points_in: NDArray[np.float32],
    points_out: NDArray[np.float32],
):
    """
    Computes an affine transformation from points_in to points_out using least squares.
    
    Args:
        points_in (NDArray[np.float32]): Array of input coordinates.
        points_out (NDArray[np.float32]): Array of target coordinates.
        
    Returns:
        NDArray: Computed affine transformation matrix.
    """
    # transform to homogenous coordinates
    points_in = np.hstack((points_in, np.ones((len(points_in), 1))), dtype=np.float32)
    points_out = np.hstack(
        (points_out, np.ones((len(points_out), 1))),
        dtype=np.float32,
    )

    # solve the least-squares problem A.T@Ax = A.Tb
    resulting = np.linalg.solve(points_in.T @ points_in, points_in.T @ points_out)

    return np.round(resulting.T, decimals=5)


def transform_points(points, matrix):
    """
    Transforms points using a given 3x3 transformation matrix.
    
    Args:
        points (NDArray): Points to transform.
        matrix (NDArray): 3x3 transformation matrix.
        
    Returns:
        NDArray: Transformed coordinates.
    """
    # homogneous coordinates
    points = np.hstack((points, np.ones((len(points), 1))), dtype=np.float32)

    # transform the points
    points = (matrix @ points.T).T
    points = points[:, :2] / points[:, 2, np.newaxis]
    return points


def get_keypoints(image, filtering=False, sigma=3):
    """
    Returns the keypoints of the image using the SIFT algorithm.
    
    Args:
        image (np.ndarray): Input image.
        filtering (bool): Optional flag to apply Gaussian filtering.
        sigma (float): Sigma for Gaussian filtering if enabled.
        
    Returns:
        tuple: (keypoints array, descriptors array)
    """
    # apply an optional smoothing to reduce the amount of found keypoints
    if filtering:
        image = gaussian_filter(image, sigma=sigma)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    # transform keypoints to numpy array
    keypoints = np.array([list(keypoints[i].pt) for i in range(len(keypoints))])

    # return the keypoints and the descriptors
    return keypoints, descriptors


def intersect2d(array1, array2):
    """
    Calculates the intersection over the rows of two arrays.
    
    Args:
        array1 (NDArray): First array.
        array2 (NDArray): Second array.
        
    Returns:
        NDArray: Array containing intersecting rows.
    """
    # tests which entries are the same
    test = array1[:, None] == array2  # needs [:, None] to induce broadcasting

    # selects only the entries, for which the
    return array2[np.all(test.mean(0) > 0, axis=1)]


def matching(descriptors_1, descriptors_2, max_ratio=0.7, cross_checking=True):
    """
    Matches the descriptors against each other.
    Returns the best match for each descriptor, if it is significant.
    The significance is defined by the max_ratio: distance_1 / distance_2 < max_ratio.
    Optional cross-checking of matches.
    
    Args:
        descriptors_1 (NDArray): Descriptors from the first image.
        descriptors_2 (NDArray): Descriptors from the second image.
        max_ratio (float): Maximum ratio between nearest neighbor distances for significance testing.
        cross_checking (bool): If True, validates matches bidirectionally.
        
    Returns:
        NDArray: Array of match indices for descriptors.
    """
    # calculate the distance between all pairs
    distance_matrix = cdist(descriptors_1, descriptors_2)

    # get the first and second match (only sorts up to the kth smallest entry, in O(n + k log(k)))
    matches, distances = (
        np.argpartition(distance_matrix, 2, axis=1)[:, :2],
        np.partition(distance_matrix, 2, axis=1)[:, :2],
    )
    if cross_checking:
        matches2, distances2 = (
            np.argpartition(distance_matrix.T, 2, axis=1)[:, :2],
            np.partition(distance_matrix.T, 2, axis=1)[:, :2],
        )

    # get the ratios
    ratios = distances[:, 0] / distances[:, 1]
    if cross_checking:
        ratios2 = distances2[:, 0] / distances2[:, 1]

    # cutoff at max_ratio
    mask = ratios < max_ratio
    if cross_checking:
        mask2 = ratios2 < max_ratio

    # produce the final matches
    final_matches = np.stack(
        (np.arange(descriptors_1.shape[0])[mask], matches[mask, 0]),
    ).T
    if cross_checking:
        final_matches2 = np.stack(
            (np.arange(descriptors_2.shape[0])[mask2], matches2[mask2, 0]),
        ).T

    # cross_checking
    if cross_checking:
        # return the intersection of the two matches arrays (invert final_matches2 to point in the same direction)
        final_matches = intersect2d(final_matches, final_matches2[:, ::-1])

    return final_matches



def compute_fundamental_matrix(points1, points2):
    """Compute the fundamental matrix given homogeneous point correspondences.

    points1 and points2 must have shape (N, 3), where each row is (x, y, 1).
    This implementation uses the algebraic 8-point SVD solution and enforces
    the rank-2 constraint manually.
    """
    points1 = np.asarray(points1, dtype=np.float64)
    points2 = np.asarray(points2, dtype=np.float64)

    if points1.shape[0] < 8 or points2.shape[0] < 8:
        raise ValueError("At least 8 point correspondences are required.")
    if points1.shape != points2.shape or points1.shape[1] != 3:
        raise ValueError("points1 and points2 must both have shape (N, 3).")

    A = []
    for p1, p2 in zip(points1, points2):
        x1, y1, _ = p1
        x2, y2, _ = p2

        A.append([
            x2 * x1,
            x2 * y1,
            x2,
            y2 * x1,
            y2 * y1,
            y2,
            x1,
            y1,
            1.0
        ])

    A = np.asarray(A, dtype=np.float64)

    # Solve A f = 0 with SVD. The solution is the last row of V^T.
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank(F) = 2 by setting the smallest singular value to zero.
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0.0
    F_rank2 = U @ np.diag(S) @ Vt

    if abs(F_rank2[-1, -1]) > 1e-12:
        F_rank2 = F_rank2 / F_rank2[-1, -1]
    else:
        norm = np.linalg.norm(F_rank2)
        if norm > 0:
            F_rank2 = F_rank2 / norm

    return F_rank2

def compute_fundamental_matrix_normalized(points1, points2):
    """
    Normalize points, compute the fundamental matrix, then denormalize it.

    Parameters
    ------------
    points1, points2 - arrays with shape (N, 2)
    """
    points1 = np.asarray(points1, dtype=np.float64)
    points2 = np.asarray(points2, dtype=np.float64)

    if points1.shape[0] < 8 or points2.shape[0] < 8:
        raise ValueError("At least 8 point correspondences are required.")
    if points1.shape != points2.shape or points1.shape[1] != 2:
        raise ValueError("points1 and points2 must both have shape (N, 2).")

    def normalize(points):
        centroid = np.mean(points, axis=0)
        shifted = points - centroid

        distances = np.sqrt(np.sum(shifted ** 2, axis=1))
        mean_distance = np.mean(distances)

        if mean_distance < 1e-12:
            raise ValueError("Cannot normalize points with near-zero spread.")

        scale = np.sqrt(2) / mean_distance

        T = np.array([
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        points_h = np.hstack((points, np.ones((points.shape[0], 1))))
        points_norm = (T @ points_h.T).T

        return points_norm, T

    points1_norm, T1 = normalize(points1)
    points2_norm, T2 = normalize(points2)

    F_norm = compute_fundamental_matrix(points1_norm, points2_norm)

    # Denormalization: x2^T F x1 = 0, with x1_norm = T1 x1 and x2_norm = T2 x2
    F = T2.T @ F_norm @ T1

    if abs(F[-1, -1]) > 1e-12:
        F = F / F[-1, -1]
    else:
        norm = np.linalg.norm(F)
        if norm > 0:
            F = F / norm

    return F


def get_max_expected_disparity(points1: np.ndarray, points2: np.ndarray) -> int:
    """
    Calculates the maximum expected horizontal disparity from matched points.

    It computes the absolute horizontal difference for each matched pair,
    takes the 99th percentile of these disparities, and rounds up.

    Args:
        points1 (np.ndarray): A (N, 2) NumPy array of (x, y) coordinates for
                              matched keypoints in the first image.
        points2 (np.ndarray): A (N, 2) NumPy array of (x, y) coordinates for
                              matched keypoints in the second image.

    Returns:
        int: The estimated maximum expected disparity, rounded up to the nearest integer.
    """
    points1 = np.asarray(points1, dtype=np.float64)
    points2 = np.asarray(points2, dtype=np.float64)

    if points1.shape != points2.shape or points1.shape[1] != 2:
        raise ValueError("points1 and points2 must both have shape (N, 2).")

    disparities = np.abs(points1[:, 0] - points2[:, 0])
    max_disparity = int(np.ceil(np.percentile(disparities, 99)))

    return max(1, max_disparity)


def compute_disparity_map(img_fixed: np.ndarray, img_search: np.ndarray, window_size: int, max_disparity: int, direction: str = "L->R") -> np.ndarray:
    """
    Computes the disparity map manually using Sum of Absolute Differences (SAD).

    direction:
        "L->R": for a pixel (y, x) in the left image, search at x - d in the right image.
        "R->L": for a pixel (y, x) in the right image, search at x + d in the left image.
    """
    img_fixed = np.asarray(img_fixed, dtype=np.float32)
    img_search = np.asarray(img_search, dtype=np.float32)

    if img_fixed.ndim == 3:
        img_fixed = cv2.cvtColor(img_fixed, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    if img_search.ndim == 3:
        img_search = cv2.cvtColor(img_search, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    h, w = img_fixed.shape
    half = window_size // 2

    disparity = np.zeros((h, w), dtype=np.float32)

    direction = direction.upper()
    if direction not in ["L->R", "R->L"]:
        raise ValueError("direction must be 'L->R' or 'R->L'.")

    for y in tqdm(range(half, h - half)):
        if direction == "L->R":
            x_range = range(half + max_disparity, w - half)
        else:
            x_range = range(half, w - half - max_disparity)

        for x in x_range:
            fixed_patch = img_fixed[y - half:y + half + 1, x - half:x + half + 1]

            best_sad = np.inf
            best_d = 0

            for d in range(max_disparity + 1):
                if direction == "L->R":
                    xs = x - d
                else:
                    xs = x + d

                search_patch = img_search[y - half:y + half + 1, xs - half:xs + half + 1]
                sad = np.sum(np.abs(fixed_patch - search_patch))

                if sad < best_sad:
                    best_sad = sad
                    best_d = d

            disparity[y, x] = best_d

    return disparity


def cross_check_disparities(disparity_L: np.ndarray, disparity_R: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Performs left-right cross-checking.

    A left disparity D_L(y, x) is valid if the corresponding right pixel
    x_right = x - D_L(y, x) has a similar disparity in D_R.
    """
    disparity_L = np.asarray(disparity_L, dtype=np.float32)
    disparity_R = np.asarray(disparity_R, dtype=np.float32)

    h, w = disparity_L.shape
    checked = np.zeros_like(disparity_L, dtype=np.float32)

    for y in range(h):
        for x in range(w):
            d_left = disparity_L[y, x]

            if d_left <= 0:
                continue

            x_right = int(round(x - d_left))

            if 0 <= x_right < w:
                d_right = disparity_R[y, x_right]

                if abs(float(d_left) - float(d_right)) <= threshold:
                    checked[y, x] = d_left

    return checked


def estimate_depth(disparity_map: np.ndarray, focal_length_pixels: float, baseline_meters: float) -> np.ndarray:
    """
    Estimates scene depth from a disparity map using Z = f * B / d.
    Invalid disparity values <= 0 are stored as NaN.
    """
    disparity_map = np.asarray(disparity_map, dtype=np.float32)
    depth_map = np.zeros_like(disparity_map, dtype=np.float32)

    valid = disparity_map > 0

    depth_map[valid] = (focal_length_pixels * baseline_meters) / disparity_map[valid]
    depth_map[~valid] = np.nan

    return depth_map


def apply_mrf(depth_map: np.ndarray, max_disparity: int, lambda_: int = 3) -> np.ndarray:
    """
    Apply an MRF/Potts-style smoothing to a map.

    If gco-wrapper is available, this uses graph cut with unary and pairwise costs.
    If it is not available, it uses a simple iterative ICM fallback with the same
    unary and pairwise idea. This keeps the implementation usable on systems where
    gco-wrapper cannot be installed.
    """
    input_map = np.asarray(depth_map, dtype=np.float32)
    valid = np.isfinite(input_map) & (input_map > 0)

    labels = np.arange(0, max_disparity + 1, dtype=np.float32)

    # Convert invalid values to 0 for unary-cost calculation.
    working_map = np.nan_to_num(input_map, nan=0.0, posinf=0.0, neginf=0.0)

    if gco is not None:
        unary = _calc_unary_costs(working_map, labels).astype(np.int32)
        pairwise = _calc_binary_costs(labels, lambda_).astype(np.int32)

        # Strongly prefer label 0 for invalid pixels.
        unary[~valid, :] = 100000
        unary[~valid, 0] = 0

        try:
            labels_result = gco.cut_grid_graph(unary, pairwise)
            smoothed = labels[labels_result].reshape(input_map.shape).astype(np.float32)
            smoothed[~valid] = np.nan
            return smoothed
        except Exception:
            # Fall back to ICM if the installed gco API differs.
            pass

    # Manual ICM fallback: unary = absolute difference from initial value,
    # pairwise = Potts penalty if neighboring labels differ.
    smoothed = np.round(working_map).astype(np.float32)
    h, w = smoothed.shape

    for _ in range(3):
        new_smoothed = smoothed.copy()

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if not valid[y, x]:
                    continue

                neighbors = [
                    smoothed[y - 1, x],
                    smoothed[y + 1, x],
                    smoothed[y, x - 1],
                    smoothed[y, x + 1],
                ]

                best_label = smoothed[y, x]
                best_cost = np.inf

                for label in labels:
                    unary_cost = abs(label - working_map[y, x])
                    pairwise_cost = 0.0

                    for neighbor in neighbors:
                        if label != neighbor:
                            pairwise_cost += lambda_

                    total_cost = unary_cost + pairwise_cost

                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_label = label

                new_smoothed[y, x] = best_label

        smoothed = new_smoothed

    smoothed[~valid] = np.nan
    return smoothed


def _calc_unary_costs(depth_map: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """
    Calculate unary costs as absolute difference between the current value
    and every proposed label.
    """
    depth_map = np.asarray(depth_map, dtype=np.float32)
    depths = np.asarray(depths, dtype=np.float32)

    costs = np.abs(depth_map[:, :, np.newaxis] - depths[np.newaxis, np.newaxis, :])

    return costs


def _calc_binary_costs(depths: np.ndarray, lambda_: int) -> np.ndarray:
    """
    Calculate Potts-model binary costs.
    Cost is 0 for equal labels and lambda_ for different labels.
    """
    depths = np.asarray(depths)

    costs = np.zeros((len(depths), len(depths)), dtype=np.int32)

    for i in range(len(depths)):
        for j in range(len(depths)):
            if i != j:
                costs[i, j] = lambda_

    return costs


def extract_surface_marching_squares(depth_map: np.ndarray, isovalue: float) -> list:
    """
    Implement the marching squares algorithm to extract isosurfaces from the depth map.
    Ambiguous saddle cases are resolved by checking the cell center.
    """
    depth_map = np.asarray(depth_map, dtype=np.float32)
    h, w = depth_map.shape

    segments = []

    def interp(p1, p2, v1, v2):
        if not np.isfinite(v1) or not np.isfinite(v2):
            return None
        if abs(v2 - v1) < 1e-12:
            t = 0.5
        else:
            t = (isovalue - v1) / (v2 - v1)
        t = np.clip(t, 0.0, 1.0)
        return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))

    for y in range(h - 1):
        for x in range(w - 1):
            # corners: top-left, top-right, bottom-right, bottom-left
            values = [
                depth_map[y, x],
                depth_map[y, x + 1],
                depth_map[y + 1, x + 1],
                depth_map[y + 1, x],
            ]

            if not np.all(np.isfinite(values)):
                continue

            points = [
                (x, y),
                (x + 1, y),
                (x + 1, y + 1),
                (x, y + 1),
            ]

            above = [v >= isovalue for v in values]

            edge_points = []

            # Edge 0: top, 1: right, 2: bottom, 3: left
            edges = [
                (0, 1),
                (1, 2),
                (3, 2),
                (0, 3),
            ]

            for a, b in edges:
                if above[a] != above[b]:
                    p = interp(points[a], points[b], values[a], values[b])
                    if p is not None:
                        edge_points.append(p)

            if len(edge_points) == 2:
                segments.append((edge_points[0], edge_points[1]))

            elif len(edge_points) == 4:
                center_value = sum(values) / 4.0

                if center_value >= isovalue:
                    segments.append((edge_points[0], edge_points[3]))
                    segments.append((edge_points[1], edge_points[2]))
                else:
                    segments.append((edge_points[0], edge_points[1]))
                    segments.append((edge_points[2], edge_points[3]))

    return segments