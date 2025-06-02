import cv2
import numpy as np
import math

def horizon_overlay(frame):
    """
    Detects the horizon line using vanishing points and overlays only the final
    RANSAC-based horizon line on the frame (no clutter).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)
    if lines is None:
        return frame

    points1, points2 = [], []
    for line in lines:
        rho, theta = line[0]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        points1.append(pt1)
        points2.append(pt2)

    selected_points1 = []
    selected_points2 = []
    for i in range(len(points1)):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        if x2 != x1:
            slope = (y2 - y1) / (x2 - x1)
            slope_in_deg = math.degrees(math.atan(slope))
            if not (87 < abs(slope_in_deg) < 93 or abs(slope_in_deg) < 5):
                selected_points1.append(points1[i])
                selected_points2.append(points2[i])
                cv2.line(frame, points1[i], points2[i], (0, 255, 255), 1)  # visualize used lines

    vanishing_points = []
    for i in range(len(selected_points1)):
        x1, y1 = selected_points1[i]
        x2, y2 = selected_points2[i]
        for j in range(i + 1, len(selected_points1)):
            x3, y3 = selected_points1[j]
            x4, y4 = selected_points2[j]

            A = np.array([[y1 - y2, x2 - x1],
                          [y3 - y4, x4 - x3]])
            b = np.array([(x2 - x1) * y1 - (y2 - y1) * x1,
                          (x4 - x3) * y3 - (y4 - y3) * x3])
            det = np.linalg.det(A)
            if abs(det) > 1e-10:
                pt = np.linalg.solve(A, b)
                vanishing_points.append((int(pt[0]), int(pt[1])))

    # Apply RANSAC to filter out unnecessary intersection points to get vanishing points
    horizon_point, inliers = ransac_average(vanishing_points)

    # Draw inlier vanishing points
    for pt in inliers:
        if 0 <= pt[0] < frame.shape[1] and 0 <= pt[1] < frame.shape[0]:
            cv2.circle(frame, tuple(pt.astype(int)), 4, (0, 0, 255), -1)  # Red dot

    # Draw horizon line
    if horizon_point != (0, 0):
        y = horizon_point[1]
        cv2.line(frame, (0, y), (frame.shape[1], y), (0, 255, 0), 2)  # Green line


    return frame, y


def ransac_average(points, num_iterations=100, threshold=10.0):
    """
    Applies RANSAC to find the best average point from a set of points.
    Returns both the best average and its inliers.
    """
    points = np.array(points)
    best_avg_point = None
    best_inliers = []

    if len(points) < 2:
        return (0, 0), []

    for _ in range(num_iterations):
        sample_indices = np.random.choice(len(points), size=2, replace=False)
        sample_points = points[sample_indices]
        avg_point = np.mean(sample_points, axis=0)
        distances = np.linalg.norm(points - avg_point, axis=1)
        inliers = points[distances < threshold]
        if len(inliers) > len(best_inliers):
            best_avg_point = np.mean(inliers, axis=0)
            best_inliers = inliers

    try:
        return tuple(best_avg_point.astype(int)), best_inliers
    except:
        return (0, 0), []
