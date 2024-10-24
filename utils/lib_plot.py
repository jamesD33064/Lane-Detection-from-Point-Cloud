
import numpy as np
import open3d
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from .lib_geo_trans import transXYZ, rotx, roty, rotz, world2pixel
from .lib_cloud_proc import downsample
from matplotlib import gridspec


def plot_cloud_2d3d(xyza, figsize=(16, 8), title='', print_time=True):
    ''' Plot two figures for a point cloud: Left is 2d; Right is 3d '''
    t0 = time.time()
    fig = plt.figure(figsize=figsize)
    if 1:  # use "gridspec" to set display size
        gs = gridspec.GridSpec(2, 8)
        ax1 = fig.add_subplot(gs[:, 0:3])
        plot_cloud_2d(xyza, ax=ax1, title=title +
                      "\n(Number of points: {})".format(xyza.shape[0]))
        ax2 = fig.add_subplot(gs[:, 3:])
        plot_cloud_3d(xyza, ax=ax2)

    else:  # use "subplot" (However, this method cannot set display size)
        plt.subplot(1, 2, 1)
        plot_cloud_2d(xyza, ax=plt.gca(), title=title)
        plt.subplot(1, 2, 2)
        plot_cloud_3d(xyza, ax=plt.gca())

    fig.tight_layout()

    if print_time:
        print("Time cost of plotting 2D/3D point cloud = {:.2f} seconds".format(
            time.time() - t0))
    return ax1, ax2


def plot_cloud_2d(xyza, figsize=(8, 6), title='', ax=None):
    ''' Plot point cloud projected on x-y plane '''

    # Set figure
    if not ax:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    ax.set_aspect('equal')
    # xyza=downsample(xyza, voxel_size=0.1) # BE CAUSIOUS OF THIS !!!

    # Set color
    red = xyza[:, -1]
    green = np.zeros_like(red)
    blue = 1 - red
    color = np.column_stack((red, green, blue))

    # Set position
    x = xyza[:, 0]
    y = xyza[:, 1]
    print(xyza.shape)

    # Plot
    plt.scatter(x, y, c=color, marker='.', linewidths=1)
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title(title, fontsize=16)
    plt.axis('on')


def plot_cloud_3d(xyza, figsize=(12, 12), title='', ax=None):
    ''' Project 3d point cloud onto 2d image, and display'''

    # Create figure axes
    if not ax:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    num_points = xyza.shape[0]

    # Camera intrinsics
    w, h = 640, 480
    camera_intrinsics = np.array([
        [w, 0, w/2],
        [0, h, h/2],
        [0, 0,   1]
    ], dtype=np.float32)

    # Set view angle
    X, X, Z, ROTX, ROTY, ROTZ = 0, 0, 0, 0, 0, 0
    X, Y, Z = -20, -68, 238
    ROTZ = np.pi/2
    ROTY = -np.pi/2.8
    T_world_to_camera = transXYZ(x=X, y=Y, z=Z).dot(
        rotz(ROTZ)).dot(rotx(ROTX)).dot(roty(ROTY))
    T_cam_to_world = np.linalg.inv(T_world_to_camera)

    # Transform points' world positions to image pixel positions
    p_world = xyza[:, 0:3].T
    p_image = world2pixel(p_world, T_cam_to_world, camera_intrinsics)
    # to int, so it cloud be plot onto image
    p_image = np.round(p_image).astype(int)

    # Put each point onto image
    zeros, ones = np.zeros((h, w)), np.ones((h, w))
    color = np.zeros((h, w, 3))
    for i in range(num_points):  # iterate through all points
        x, y, a = p_image[0, i], p_image[1, i], xyza[i, -1]
        u, v = y, x  # flip direction to match the plt plot
        if w > u >= 0 and h > v >= 0:
            color[v][u][0] = max(color[v][u][0], a)
            color[v][u][2] = 1 - color[v][u][0]

    # Show
    ax.imshow(color)
    plt.axis('off')


# def plot_3d_cloud(cloud):

#     ''' Plot 3d points using Axes3D '''

#     if isinstance(cloud, open3d.PointCloud):
#         xyz = np.asarray(cloud.points)
#     else:
#         xyz = cloud[:, 0:3]

#     fig = plt.figure()
#     ax = Axes3D(fig)

#     x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
#     ax.scatter(x, y, z, marker='.', linewidth=1)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')

def plot_2d_line_from_3d(xyza, head_pts, tail_pts, figsize=(12, 12), title='', ax=None):
    ''' 
    將 3D 點雲投影到 2D 圖像上，並顯示直線 
    輸入:
        xyza: 3D 點雲數據，形狀為 (n, 4)
        head_pts: 直線頭部座標 (x, y, z)
        tail_pts: 直線尾部座標 (x, y, z)
        figsize: 圖像大小
        title: 圖像標題
        ax: matplotlib 的 axes，默認為 None
    '''

    # 如果沒有提供 ax，則創建 figure 和 axes
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    
    num_points = xyza.shape[0]

    # 相機內參數
    w, h = 640, 480
    camera_intrinsics = np.array([
        [w, 0, w/2],
        [0, h, h/2],
        [0, 0,   1]
    ], dtype=np.float32)

    # 設置相機的視角與位置
    X, Y, Z = -20, -68, 238
    ROTZ = np.pi / 2
    ROTY = -np.pi / 2.8
    T_world_to_camera = transXYZ(x=X, y=Y, z=Z).dot(
        rotz(ROTZ)).dot(rotx(0)).dot(roty(ROTY))
    T_cam_to_world = np.linalg.inv(T_world_to_camera)

    # 轉換點雲到 2D 像素座標
    p_world = xyza[:, 0:3].T
    p_image = world2pixel(p_world, T_cam_to_world, camera_intrinsics)
    p_image = np.round(p_image).astype(int)

    # 構建影像
    color = np.zeros((h, w, 3))  # 黑色背景
    for i in range(num_points):
        x, y, a = p_image[0, i], p_image[1, i], xyza[i, -1]
        u, v = y, x  # 翻轉 x, y 方向以匹配 plt 的顯示
        if w > u >= 0 and h > v >= 0:
            color[v][u][0] = max(color[v][u][0], a)  # 紅色通道
            color[v][u][2] = 1 - color[v][u][0]      # 藍色通道

    # 將頭部和尾部座標轉換為 2D 像素座標
    p_line_world = np.vstack((head_pts, tail_pts)).T
    p_line_image = world2pixel(p_line_world, T_cam_to_world, camera_intrinsics)
    p_line_image = np.round(p_line_image).astype(int)

    # 提取頭尾的 2D 座標，並對調 u 和 v，以匹配圖像軸
    v_head, u_head = p_line_image[0, 0], p_line_image[1, 0]
    v_tail, u_tail = p_line_image[0, 1], p_line_image[1, 1]

    # 在圖像上繪製直線
    line = [[u_head, u_tail], [v_head, v_tail]]
    ax.plot(line[0], line[1], color='y', linewidth=2)

    # 顯示圖像
    ax.imshow(color)
    ax.axis('off')  # 關閉座標軸
    ax.set_title(title)
