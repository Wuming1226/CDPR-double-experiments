#! /usr/bin/env python3

import time
import numpy as np
import math
import rospy
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from cdpr import CDPR
from jacobian import get_jacobian
from generate_traject import smooth_p2p

folder = 'src/double_cdpr/data/beta/'
file_order = ''
pose_ref_save_path = folder + 'pose_ref' + file_order + '.txt'
pose_save_path = folder + 'pose' + file_order + '.txt'
# cable_length_save_path = folder + 'cable_length' + file_order + '.txt'
motor_velo_save_path = folder + 'motor_velo_' + file_order + '.txt'


if __name__ == "__main__":

    cdpr = CDPR()

    T = 0.1     # control period
    rate = rospy.Rate(1/T)

    x_r_list, y_r_list, z_r_list = [], [], []
    x_list, y_list, z_list = [], [], []
    # cl1_list, cl2_list, cl3_list, cl4_list = [], [], [], []
    pose_list = np.empty((0, 4))
    pose_ref_list = np.empty((0, 4))
    # cable_length_ref_list = np.empty((0, 4))
    # cable_length_list = np.empty((0, 4))
    motor_velo_list = np.empty((0, 4))

    time.sleep(2)

    # traject = np.loadtxt("trajectory_planned.txt")
    start_point = np.array(cdpr.get_moving_platform_pose()[0:3])
    print(start_point)
    end_point = start_point + np.array([0, 0, 0.1])
    traject = smooth_p2p([start_point, end_point, start_point], [10, 10], np.inf, 0.05)

    tighten_flag = True

    # ---------------------- main loop ----------------------

    # cdpr.pretighten(True, True, True, True)
    cdpr.init_cable_length(True, True, True, True)

    cnt = 0
    lst_err = 0

    # ---------------------- main loop ----------------------

    while not rospy.is_shutdown() and cnt < len(traject):

        print('-----------------------------------------------')
        print('                   run: {}'.format(cnt))

        start_time = time.time()

        # 参考数值（所有数据均在基座坐标系下）
        pose_ref = np.append(traject[cnt], 0)
        print('pose_ref: {}'.format(pose_ref))

        if cnt == len(traject) - 1:     # 防溢出
            pose_ref_next = np.append(traject[cnt], 0)
        else:
            pose_ref_next = np.append(traject[cnt + 1], 0)
        print('pose_ref_next: {}'.format(pose_ref_next))

        cnt += 1

        # 实际数值
        x, y, z, orient = cdpr.get_moving_platform_pose()
        phi = R.from_quat(orient).as_euler('ZYX', degrees=False)[0]
        pose = np.array([x, y, z, phi])
        print('pose: {}'.format(pose))
        # cable_length = cdpr.get_cable_length()
        # print('cable length: {}'.format(cable_length))

        # 位姿误差
        pose_err = pose_ref.reshape(-1, 1) - pose.reshape(-1, 1)
        print('pose err: {}'.format(pose_err))

        # 滑模控制器
        eps = np.diag(np.array([0.002, 0.002, 0.002, 0.002]))
        k = np.diag(np.array([1.5, 1.5, 1.5, 1.5]))
        velo_task = (pose_ref_next.reshape(-1, 1) - pose_ref.reshape(-1, 1)) / T + eps @ np.sign(pose_err) + k @ pose_err  # control law
        print('velo_task: {}'.format(velo_task))

        # 逆运动学
        J = get_jacobian(cdpr.a_matrix, cdpr.b_matrix, pose[0:3], orient)
        velo_joint = J @ velo_task
        velo_joint = velo_joint.reshape(4, )
        print('veloJoint: {}'.format(velo_joint))

        # convert linear velocities to velocities of motors
        velo_motor = velo_joint / (0.04*math.pi)      # 0.04 is diameter of the coil

        # set cable velocity in joint space
        velo_limit = 3.5
        for i, vel in enumerate(velo_motor):
            if np.abs(vel) > velo_limit:      # velocity limit
                velo_motor[i] = velo_limit * np.sign(vel)

        cdpr.set_motor_velo(velo_motor[0], velo_motor[1], velo_motor[2], velo_motor[3])
        print('motor velo: {}, {}, {}, {}'.format(velo_motor[0], velo_motor[1], velo_motor[2], velo_motor[3]))

        x_r_list.append(pose_ref[0])
        y_r_list.append(pose_ref[1])
        z_r_list.append(pose_ref[2])

        x_list.append(pose[0])
        y_list.append(pose[1])
        z_list.append(pose[2])

        # cl1_list.append(cable_length[0])
        # cl2_list.append(cable_length[1])
        # cl3_list.append(cable_length[2])
        # cl4_list.append(cable_length[3])

        # data 
        pose_list = np.vstack((pose_list, pose))
        pose_ref_list = np.vstack((pose_ref_list, pose_ref))
        # cable_length_list = np.row_stack((cable_length_list, cable_length))
        # length_controller_list = np.row_stack((length_controller_list, veloJoint1))
        motor_velo_list = np.vstack((motor_velo_list, velo_motor))

        np.savetxt(pose_ref_save_path, pose_ref_list)
        np.savetxt(pose_save_path, pose_list)
        # np.savetxt(cable_length_save_path, cable_length_list)
        # np.savetxt(length_controller_save_path, length_controller_list)
        np.savetxt(motor_velo_save_path, motor_velo_list)
        print('data saved.')

        end_time = time.time()
        print("loop time: {}".format(end_time - start_time))

        rate.sleep()

    cdpr.set_motor_velo(0, 0, 0, 0)

    # calculate error
    x_e = np.array(x_r_list) - np.array(x_list)
    y_e = np.array(y_r_list) - np.array(y_list)
    z_e = np.array(z_r_list) - np.array(z_list)
    err_arr = np.sqrt(x_e ** 2 + y_e ** 2 + z_e ** 2)
    print("\n\n-----------------------------")
    print("mean tracking error: {}".format(np.mean(err_arr)))
    print("max tracking error: {}".format(np.max(err_arr)))

    # plot
    fig = plt.figure(1)
    x_plot = fig.add_subplot(4, 2, 1)
    y_plot = fig.add_subplot(4, 2, 3)
    z_plot = fig.add_subplot(4, 2, 5)
    # c1_plot = fig.add_subplot(4, 2, 2)
    # c2_plot = fig.add_subplot(4, 2, 4)
    # c3_plot = fig.add_subplot(4, 2, 6)
    # c4_plot = fig.add_subplot(4, 2, 8)

    x_plot.plot(x_r_list)
    x_plot.plot(x_list)
    y_plot.plot(y_r_list)
    y_plot.plot(y_list)
    z_plot.plot(z_r_list)
    z_plot.plot(z_list)

    # c1_plot.plot(cl1_list)
    # c2_plot.plot(cl2_list)
    # c3_plot.plot(cl3_list)
    # c4_plot.plot(cl4_list)

    x_plot.set_ylabel('x')
    y_plot.set_ylabel('y')
    z_plot.set_ylabel('z')

    plt.ioff()
    plt.show()

    # save trajectory datas
    np.savetxt(pose_ref_save_path, pose_ref_list)
    np.savetxt(pose_save_path, pose_list)
    # np.savetxt(cable_length_save_path, cable_length_list)
    np.savetxt(motor_velo_save_path, motor_velo_list)
    print('data saved.')

