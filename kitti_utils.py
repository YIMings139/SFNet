from __future__ import absolute_import, division, print_function
import os
import numpy as np
from collections import Counter


def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        # 文件每行的形式如：
        # calib_time: 09-Jan-2012 13: 57:47
        for line in f.readlines():
            key, value = line.split(':', 1)  # 只切分一次。
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # float_chars.issuperset(value) 的含义是检查字符串 value 中的所有字符是否都包含在 float_chars 这个字符集合中。
                # 看这个superset，说明float_chars应该比value更大才为true。
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                    # 得到data字典，包含一系列参数。
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n - 1) + colSub - 1


def generate_depth_map(calib_dir, velo_filename, cam=2, vel_depth=False):
    """Generate a depth map from velodyne data
    """
    # "Velodyne data" 通常指的是由 Velodyne 激光雷达传感器收集的激光雷达数据。

    # load calibration files
    # 这段代码的主要目的是加载相机和激光雷达之间的标定参数，并将其转换为适用于数据处理的格式。
    # 这些标定文件通常包含了相机的内部参数、外部参数、旋转矩阵、平移矩阵等信息。

    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    # 得到了相机到相机之间的参数。
    # ##############################################################################
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    # 同理，得到了激光雷达到相机之间的参数。
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    # np.hstack是水平拼接。
    # 将 R 这一行变为3x3的矩阵，并将 T 这一行 原本为1x3的向量通过[..., np.newaxis]这个方法扩展成与R相同维度的矩阵，最终变为3x4。
    # 是将 T这一行转置后接在R这个方阵的后边。因此T的列数要与R的行数一致
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
    # 这个np.vstack是竖直拼接。

    # ############### 这里就得到了4X4的雷达到相机之间的参数矩阵 #####################

    # get image shape
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)
    # [::-1]是取倒序,并用astype(np.int32)将之前的科学计数法转变为正常十进制数。
    # 得到了图像的高宽。

    # compute projection matrix velodyne->image plane
    # 计算从 Velodyne 点云到图像平面的投影矩阵
    R_cam2rect = np.eye(4)  # 单位阵。
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0' + str(cam)].reshape(3, 4)  # cam是第几个摄像头，这里是用的第二个摄像头
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)  # 最终得到了 3x4 的转换矩，P_velo2im。

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_filename)
    # 导入bin文件，得到一个numpy数组，为点云数据，形状为[-1,4]。
    velo = velo[velo[:, 0] >= 0, :]
    # 进行筛选，选择每一行第一列的数大于零的这个整行。大于零代表向前。

    # project the points to the camera

    velo_pts_im = np.dot(P_velo2im, velo.T).T  # [3x4] 点乘 [4,-1] 得到 [3，-1] 再转置得到 [-1,3]

    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]
    # 前两列除以第三列

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # ################ 这段代码的作用是检查点云数据中的点是否在图像的边界内，并将边界外的点过滤掉。 #############
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1

    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1

    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    # 对第一列和第二列进行逐元素比较，如果都大于零就输出True，val_inds的形状为velo_pts_im的行数。
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]
    # 只得到True的对象行数。

    # project to image
    depth = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    # Convert row, col matrix subscripts to linear indices
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]

    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth
