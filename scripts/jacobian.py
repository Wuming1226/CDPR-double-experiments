import numpy as np
from scipy.spatial.transform import Rotation as R


def get_jacobian(a_matrix, b_matrix, position, orientation):
    """
    :param a_matrix: positions of anchorAs (3×n, world frame)
    :param b_matrix: positions of anchorBs (3×n, body frame)
    :param position: position of end effector in world frame
    :param orientation: orientation of end effector in world frame
    :return: Jacobian matrix

    """
    quat_orient = R.from_quat(orientation)      # scale-last (x, y, z, w)

    # rotation
    rotated_b_matrix = quat_orient.apply(b_matrix)

    # translation
    transformed_b_matrix = rotated_b_matrix + position

    # calculate un and bn x un
    u_matrix = a_matrix - transformed_b_matrix
    u_matrix = u_matrix / np.linalg.norm(u_matrix, axis=1, keepdims=True)

    bu_matrix = np.cross(transformed_b_matrix, u_matrix)

    # J =  - |    u1       u2    ...     un   | . T
    #        | b1 x u1  b2 x u2  ...  bn x un |
    #
    #           x  y  z    r  p  y
    #   =    |   u1.T    (b1 x u1).T  |
    #        |   u2.T    (b2 x u2).T  |
    #        |   ...         ...      |
    #        |   un.T    (bn x un).T  |
    #

    jacobian = -np.hstack((u_matrix, bu_matrix))
    jacobian = jacobian[:, [0, 1, 2, 5]]

    return jacobian


if __name__ == "__main__":
    # anchors on the fixed frame (world frame)
    anchorA1Pos = np.array([1, 1, 1.5])
    anchorA2Pos = np.array([-1, 1, 1.5])
    anchorA3Pos = np.array([-1, -1, 1.5])
    anchorA4Pos = np.array([1, -1, 1.5])
    anchorAPos = np.vstack([anchorA1Pos, anchorA2Pos, anchorA3Pos, anchorA4Pos])
    print(anchorAPos)

    # anchors on the moving platform (body frame)
    anchorB1Pos = np.array([0.1, 0.1, 0.1])
    anchorB2Pos = np.array([-0.1, 0.1, 0.1])
    anchorB3Pos = np.array([-0.1, -0.1, 0.1])
    anchorB4Pos = np.array([0.1, -0.1, 0.1])
    anchorBPos = np.vstack([anchorB1Pos, anchorB2Pos, anchorB3Pos, anchorB4Pos])

    pos = np.array([0, 0.1, 0.4])
    orientation = np.array([0, 0, 0, 1])

    J = get_jacobian(anchorAPos, anchorBPos, pos, orientation)
    print(J)
    velo = np.array([0.03, 0.01, 0.01, 0.0])
    velo = np.array([0.0, 0.0, 0.0, 0.1])
    dx = J @ velo.reshape(4, )
    print(dx)

