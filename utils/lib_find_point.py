from scipy.spatial import distance, ConvexHull
from sklearn.decomposition import PCA
import numpy as np 

def get_farthest_points(cluster_data):
    """找到該分群中距離最遠的兩個點"""
    max_distance = 0
    point1, point2 = None, None
    
    # 計算所有點之間的距離
    for i in range(len(cluster_data)):
        for j in range(i + 1, len(cluster_data)):
            dist = distance.euclidean(cluster_data[i], cluster_data[j])
            if dist > max_distance:
                max_distance = dist
                point1, point2 = cluster_data[i], cluster_data[j]
    
    # return point1, point2, max_distance
    return point1, point2

def get_corner_points(cluster_data):
    """找到該分群中四個角落的點，並取長邊的兩個中點座標"""
    hull = ConvexHull(cluster_data[:, :2]) 
    hull_points = cluster_data[hull.vertices]

    min_x_point = hull_points[np.argmin(hull_points[:, 0])]
    max_x_point = hull_points[np.argmax(hull_points[:, 0])]
    min_y_point = hull_points[np.argmin(hull_points[:, 1])]
    max_y_point = hull_points[np.argmax(hull_points[:, 1])]

    min_center = (min_x_point + min_y_point) / 2
    max_center = (max_x_point + max_y_point) / 2
    return min_center, max_center

def get_PCA_points(cluster_data):
    hull = ConvexHull(cluster_data[:, :2]) 
    hull_points = cluster_data[hull.vertices]

    # 使用 PCA 來計算主要趨勢方向
    pca = PCA(n_components=2)
    pca.fit(cluster_data)
    mean = pca.mean_  # 點雲的中心
    components = pca.components_  # 特徵向量
    explained_variance = pca.explained_variance_  # 特徵值

    direction = components[0]  # 主要趨勢方向

    # 將凸包上的點投影到主要趨勢方向
    projections = np.dot(hull_points - mean, direction)

    # 計算長邊的長度
    min_proj_point = hull_points[np.argmin(projections)]
    max_proj_point = hull_points[np.argmax(projections)]
    long_edge_length = np.linalg.norm(max_proj_point - min_proj_point)

    line_start = mean - direction * (long_edge_length / 2)
    line_end = mean + direction * (long_edge_length / 2)
    return line_start, line_end


def closest_point_on_line(P1, P2, P0):
    """計算直線 P1P2 上最靠近點 P0 的點的坐標"""
    P1 = np.array(P1)
    P2 = np.array(P2)
    P0 = np.array(P0)
    
    d = P2 - P1
    v = P0 - P1
    
    t = np.dot(v, d) / np.dot(d, d)
    closest_point = P1 + t * d
    
    return closest_point
