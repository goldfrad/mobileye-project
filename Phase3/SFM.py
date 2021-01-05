from operator import itemgetter

import numpy as np


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if abs(tZ) < 10e-6:
        print('tz = ', tZ)
    elif norm_prev_pts.size == 0:
        print('no prev points')
    elif norm_prev_pts.size == 0:
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = \
            calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ)

    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))

    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []

    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)

        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)

    return corresponding_ind, np.array(pts_3D), validVec


# transformition pixels into normalized pixels using the focal length and principle point

def normalize(pts, focal, pp):
    pts_normalize = []

    for pt in pts:
        point0 = (pt[0] - pp[0]) / focal
        point1 = (pt[1] - pp[1]) / focal
        pts_normalize.append([point0, point1])

    return np.array(pts_normalize)


def unnormalize(pts, focal, pp):
    pts_unnormalize = []

    for pt in pts:
        point0 = pt[0] * focal + pp[0]
        point1 = pt[1] * focal + pp[1]
        pts_unnormalize.append([point0, point1])

    return np.array(pts_unnormalize)


def decompose(EM):
    tz = EM[2, 3]
    p0 = EM[0, 3] / tz
    p1 = EM[1, 3] / tz
    
    return EM[:3, :3], [p0, p1], tz


def rotate(pts, R):
    ptsr = []

    for pt in pts:
        res = R.dot(np.append(pt, np.array(1)))
        ptsr.append([res[0] / res[2], res[1] / res[2]])

    return ptsr


# compute the epipolar line between p and foe

def find_corresponding_points(p, norm_pts_rot, foe):
    x, y = 0, 1
    m = (foe[y] - p[y]) / (foe[x] - p[x])
    n = (p[y] * foe[x] - foe[y] * p[x]) / (foe[x] - p[x])
    list_ = [[abs((m * pts[x] + n - pts[y]) / np.sqrt(pow(m, 2) + 1)), i]
             for i, pts in enumerate(norm_pts_rot)]
    min_dist, i_min = min(list_, key=itemgetter(0))

    return i_min, norm_pts_rot[i_min]


def calc_dist(p_curr, p_rot, foe, tZ):  # Calculation of distances
    dis_x = tZ * (foe[0] - p_rot[0]) / (p_curr[0] - p_rot[0])
    dis_y = tZ * (foe[1] - p_rot[1]) / (p_curr[1] - p_rot[1])
    dX = abs(foe[0] - p_curr[0])
    dY = abs(foe[1] - p_curr[1])
    ratio = dX / (dY + dX)

    return dis_x * ratio + dis_y * (1 - ratio)
