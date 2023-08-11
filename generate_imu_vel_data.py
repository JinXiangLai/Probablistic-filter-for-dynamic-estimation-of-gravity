import numpy as np
import math
from gravity_evaluate import quaternion_matrix, angle_velocity2rotation, hat

data_folder = ''
data_file_name = data_folder + 'vel_imu_simulate.csv'


def content_gyro_acc(t, gyro, acc) -> str:
    content = 'IMU ' + str(t)
    for i in gyro:
        content += ' ' + str(i)
    for i in acc:
        content += ' ' + str(i)
    return content + '\n'


def content_vel_quat(t, v, quat) -> str:
    content = 'Novetal ' + str(t)
    for i in v:
        content += ' ' + str(i)
    for i in quat:
        content += ' ' + str(i)
    return content + '\n'


def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> numpy.allclose(quaternion_from_matrix(R, isprecise=False),
    ...                quaternion_from_matrix(R, isprecise=True))
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    # 这里的q是[qw, qx, qy, qz],将其转换为[qx, qy, qz, qw]
    q = np.array([q[1], q[2], q[3], q[0]], dtype=np.float64, copy=True)
    q = q/np.linalg.norm(q)
    return q


ax = 1
ay = 0.001
az = 9.8
sigma_ax = 0.1
sigma_ay = 0.01
sigma_az = 0.001

wx = 0.0001
wy = 0.001
wz = 0.1
sigma_wx = 0.0001
sigma_wy = 0.001
sigma_wz = 0.01


def generate_acc_data():
    a0 = ax + np.random.normal(loc=0, scale=sigma_ax)
    a1 = ay + np.random.normal(loc=0, scale=sigma_ay)
    a2 = az + np.random.normal(loc=0, scale=sigma_az)
    return np.array([a0, a1, a2])


def generate_gyro_data():
    w0 = wx + np.random.normal(loc=0, scale=sigma_wx)
    w1 = wy + np.random.normal(loc=0, scale=sigma_wy)
    w2 = wz + np.random.normal(loc=0, scale=sigma_wz)
    return np.array([w0, w1, w2])


def calculate_vel_acc(v, q, gyro, acc, delta_t, g):
    R = quaternion_matrix(q)
    v = v + R.dot(acc) * delta_t + g * delta_t
    R = R.dot(angle_velocity2rotation(gyro, delta_t))
    H = np.eye(4)
    H[:3, :3] = R
    q = quaternion_from_matrix(H)
    return v, q


def main():
    # 宏参数
    sample_time = 0.01  # 10 ms
    vel = np.array([0., 0., 0.])
    # 注意，四元素的形式
    quat = np.array([0., 0., 0., 1.])
    g = np.array([0, 0, -9.8])
    generate_data_num = 10000
    timestamp = 0

    with open(data_file_name, 'w') as file:
        initial_content = '#' + 'type ' + 'timestamp ' + 'content\n'
        file.write(initial_content)
        # 用于记录当要生成v, quat时，不再更新w, a
        # 使得算出来的v,quat使用欧拉积分是正确的
        w = np.array([0., 0., 0.])
        a = np.array([0., 0., -9.8])
        sample_freq = 3
        for i in range(generate_data_num):
            if i % sample_freq:
                # 重新采样并写入文件
                w = generate_gyro_data()
                a = generate_acc_data()
                content = content_gyro_acc(timestamp, w, a)
                file.write(content)

            # 计算速度和旋转
            vel, quat = calculate_vel_acc(vel, quat, w, a, sample_time, g)
            if i % sample_freq == 0:
                content = content_vel_quat(timestamp, vel, quat)
                file.write(content)

            timestamp += sample_time


if __name__ == '__main__':
    main()
