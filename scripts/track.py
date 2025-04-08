#! /usr/bin/env python3

import time
import numpy as np
import math
import rospy
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from cdpr import CDPR
from jacobian import get_jacobian

folder = '../data/beta/'
file_order = ''
pos_ref_save_path = folder + 'pos_ref' + file_order + '.txt'
pos_save_path = folder + 'pos' + file_order + '.txt'
# orient_ref_save_path = folder + 'orient_ref' + file_order + '.txt'
# orient_save_path = folder + 'orient' + file_order + '.txt'
cable_length_save_path = folder + 'cable_length' + file_order + '.txt'
length_controller_save_path = folder + 'length_controller' + file_order + '.txt'
velocity_controller_task_save_path = folder + 'velocity_controller_task' + file_order + '.txt'
velocity_controller_joint_save_path = folder + 'velocity_controller_joint' + file_order + '.txt'
motor_velo_save_path = folder + 'motor_velo_' + file_order + '.txt'


if __name__ == "__main__":

    cdpr = CDPR()

    T = 0.1     # control period
    rate = rospy.Rate(1/T)

    x_r_list, y_r_list, z_r_list = [], [], []
    x_list, y_list, z_list = [], [], []
    cl1_list, cl2_list, cl3_list, cl4_list = [], [], [], []
    pos_list = np.empty((0, 3))
    pos_ref_list = np.empty((0, 3))
    cable_length_ref_list = np.empty((0, 4))
    cable_length_list = np.empty((0, 4))
    length_controller_list = np.empty((0, 4))
    velocity_controller_task_list = np.empty((0, 3))
    velocity_controller_joint_list = np.empty((0, 4))
    motor_velo_list = np.empty((0, 4))

    traject = np.loadtxt("trajectory+.txt")

    tighten_flag = True

    # ---------------------- main loop ----------------------

    time.sleep(2)
    # cdpr.pretighten(True, True, True, True)
    cdpr.init_cable_length(True, True, True, True)

    cnt = 0
    lst_err = 0

    # ---------------------- main loop ----------------------

    while not rospy.is_shutdown() and cnt < len(traject):

        print('-----------------------------------------------')
        print('                   run: {}'.format(cnt))

        start_time = time.time()

        # 参考数值
        pos_ref = traject[cnt, :3]
        print('pos_ref: {}'.format(pos_ref))

        orient_ref = traject[cnt, 3:]
        print('orient_ref: {}'.format(orient_ref))

        if cnt == len(traject) - 1:     # 防溢出
            pos_ref_next = traject[cnt]
            orient_ref_next = traject[cnt]
        else:
            pos_ref_next = traject[cnt+1]
            orient_ref_next = traject[cnt+1]
        print('pos_ref_next: {}'.format(pos_ref_next))
        print('orient_ref_next: {}'.format(orient_ref_next))

        cnt += 1

        # 实际数值
        x, y, z, orient = cdpr.get_moving_platform_pose()
        pos = np.array([x, y, z])
        print('pos: {}'.format(pos))
        print('orient: {}'.format(orient))
        cable_length = cdpr.get_cable_length()
        print('cable length: {}'.format(cable_length))

        # 位置误差
        pos_err = pos_ref - pos
        print('pos err: {}'.format(pos_err))

        # 姿态误差
        orient_err = (R.from_quat(orient_ref) * R.from_quat(orient).inv()).as_rotvec()
        print('orient err: {}'.format(orient_err))

        # 控制器
        eps = 0.002
        k = 1.5
        velo_task = (pos_ref_next - pos_ref) / T + eps * np.sign(pos_err) + k * pos_err  # control law
        print('velo_task: {}'.format(velo_task))

        eps = 0.002
        k = 1.5
        ang_velo_task = ((R.from_quat(orient_ref_next).as_rotvec() - R.from_quat(pos_ref).as_rotvec()) / T +
                         eps * np.sign(orient_err) + k * orient_err)    # control law
        print('ang_velo_task: {}'.format(ang_velo_task))

        # 逆运动学
        J = get_jacobian(cdpr.a_matrix, cdpr.b_matrix, pos, orient)
        velo_joint = np.matmul(J, velo_task.reshape(3, 1))
        # velo_joint = np.matmul(J, np.vstack([velo_task.reshape(3, 1), ang_velo_task.reshape(3, 1)]))
        velo_joint = velo_joint.reshape(4, )
        print('veloJoint: {}'.format(velo_joint))

        # convert linear velocities to velocities of motors
        velo_motor = velo_joint * 60 * 10 / (0.03*math.pi)      # 10 is the gear ratio, 0.03 is diameter of the coil

        # set cable velocity in joint space
        velo_limit = 3
        for i, vel in enumerate(velo_motor):
            if np.abs(vel) > velo_limit:      # velocity limit
                velo_motor[i] = velo_limit * np.sign(vel)

        cdpr.set_motor_velo(int(velo_motor[0]), int(velo_motor[1]), int(velo_motor[2]), int(velo_motor[3]))
        print('motor velo: {}, {}, {}, {}'.format(velo_motor[0], velo_motor[1], velo_motor[2], velo_motor[3]))

        x_r_list.append(pos_ref[0])
        y_r_list.append(pos_ref[1])
        z_r_list.append(pos_ref[2])

        x_list.append(pos[0])
        y_list.append(pos[1])
        z_list.append(pos[2])

        cl1_list.append(cable_length[0])
        cl2_list.append(cable_length[1])
        cl3_list.append(cable_length[2])
        cl4_list.append(cable_length[3])

        # data 
        pos_list = np.row_stack((pos_list, pos))
        pos_ref_list = np.row_stack((pos_ref_list, pos_ref))
        cable_length_list = np.row_stack((cable_length_list, cable_length))
        # length_controller_list = np.row_stack((length_controller_list, veloJoint1))
        # velocity_controller_task_list = np.row_stack((velocity_controller_task_list, veloTask))
        # velocity_controller_joint_list = np.row_stack((velocity_controller_joint_list, veloJoint2))
        motor_velo_list = np.row_stack((motor_velo_list, velo_motor))

        np.savetxt(pos_ref_save_path, pos_ref_list)
        np.savetxt(pos_save_path, pos_list)
        np.savetxt(cable_length_save_path, cable_length_list)
        # np.savetxt(length_controller_save_path, length_controller_list)
        # np.savetxt(velocity_controller_task_save_path, velocity_controller_task_list)
        np.savetxt(velocity_controller_joint_save_path, velocity_controller_joint_list)
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
    c1_plot = fig.add_subplot(4, 2, 2)
    c2_plot = fig.add_subplot(4, 2, 4)
    c3_plot = fig.add_subplot(4, 2, 6)
    c4_plot = fig.add_subplot(4, 2, 8)

    x_plot.plot(x_r_list)
    x_plot.plot(x_list)
    y_plot.plot(y_r_list)
    y_plot.plot(y_list)
    z_plot.plot(z_r_list)
    z_plot.plot(z_list)

    c1_plot.plot(cl1_list)
    c2_plot.plot(cl2_list)
    c3_plot.plot(cl3_list)
    c4_plot.plot(cl4_list)

    x_plot.set_ylabel('x')
    y_plot.set_ylabel('y')
    z_plot.set_ylabel('z')

    plt.ioff()
    plt.show()

    # save trajectory datas
    np.savetxt(pos_ref_save_path, pos_ref_list)
    np.savetxt(pos_save_path, pos_list)
    np.savetxt(cable_length_save_path, cable_length_list)
    # np.savetxt(length_controller_save_path, length_controller_list)
    # np.savetxt(velocity_controller_task_save_path, velocity_controller_task_list)
    np.savetxt(velocity_controller_joint_save_path, velocity_controller_joint_list)
    np.savetxt(motor_velo_save_path, motor_velo_list)
    print('data saved.')

