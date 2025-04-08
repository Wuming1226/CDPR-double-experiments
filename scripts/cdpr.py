#! /usr/bin/env python3

import rospy
import time
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Int16MultiArray, Int64MultiArray

from jacobian import get_jacobian


class CDPR:

    def __init__(self):
        # ros settings
        rospy.init_node('cdpr_control', anonymous=True)

        # subscriber and publisher
        self._moving_platform_pose = PoseStamped()
        rospy.Subscriber('/vrpn_client_node/end_effector/pose', PoseStamped, self._pose_callback, queue_size=1)

        self._motor_pos = np.array([0, 0, 0, 0])
        rospy.Subscriber('motor_pos', Int64MultiArray, self._motor_pos_callback, queue_size=1)

        self._veloPub = rospy.Publisher('motor_velo', Int16MultiArray, queue_size=10)

        # origin point offset (coordinates in world frame)
        self.x_off = 0.0
        self.y_off = 0.0
        self.z_off = 0.0
        self.pos_off = np.array([self.x_off, self.y_off, self.z_off])

        # anchor positions in the world frame
        self._anchorA1 = np.array([0.342, 0.342, 0.727])
        self._anchorA2 = np.array([-0.342, 0.342, 0.727])
        self._anchorA3 = np.array([-0.342, -0.342, 0.727])
        self._anchorA4 = np.array([0.342, -0.342, 0.727])
        self.a_matrix = np.vstack([self._anchorA1, self._anchorA2, self._anchorA3, self._anchorA4])

        self._anchorB1 = np.array([0.000, 0.000, 0.000])
        self._anchorB2 = np.array([0.000, 0.000, 0.000])
        self._anchorB3 = np.array([0.000, 0.000, 0.000])
        self._anchorB4 = np.array([0.000, 0.000, 0.000])
        self.b_matrix = np.vstack([self._anchorB1, self._anchorB2, self._anchorB3, self._anchorB4])

        # initial cable lengths and motor positions
        self._ori_cable_lengths = np.array([0., 0., 0., 0.])     # 初始化的类型必须是浮点！！！

    def init_cable_length(self, cable1_flag, cable2_flag, cable3_flag, cable4_flag):
        # calculate origin cable lengths
        time.sleep(1)
        x0, y0, z0, orient0 = self.get_moving_platform_pose()
        pos0 = np.array([x0, y0, z0])

        quat_orient0 = R.from_quat(orient0)
        b_matrix = quat_orient0.apply(self.b_matrix)

        if cable1_flag:
            self._ori_cable_lengths[0] = np.linalg.norm(pos0 - self._anchorA1 - b_matrix[0, :])
        if cable2_flag:
            self._ori_cable_lengths[1] = np.linalg.norm(pos0 - self._anchorA2 - b_matrix[1, :])
        if cable3_flag:
            self._ori_cable_lengths[2] = np.linalg.norm(pos0 - self._anchorA3 - b_matrix[2, :])
        if cable4_flag:
            self._ori_cable_lengths[3] = np.linalg.norm(pos0 - self._anchorA4 - b_matrix[3, :])

    def _pose_callback(self, data):
        # if motion data is lost(999999), do not update
        if (np.abs(data.pose.position.x) > 2000 or np.abs(data.pose.position.y) > 2000
                or np.abs(data.pose.position.z) > 2000):
            pass
        else:
            # pose
            self._moving_platform_pose.pose.position.x = data.pose.position.x / 1000 - self.pos_off[0]
            self._moving_platform_pose.pose.position.y = data.pose.position.y / 1000 - self.pos_off[1]
            self._moving_platform_pose.pose.position.z = data.pose.position.z / 1000 - self.pos_off[2]
            self._moving_platform_pose.pose.orientation = data.pose.orientation

            # header
            self._moving_platform_pose.header.frame_id = data.header.frame_id
            self._moving_platform_pose.header.stamp = data.header.stamp

    def _motor_pos_callback(self, data):
        self._motor_pos = np.array(data.data)

    def set_motor_velo(self, motor1_velo, motor2_velo, motor3_velo, motor4_velo):
        motor_velo = Int16MultiArray(data=np.array([motor1_velo, motor2_velo, motor3_velo, motor4_velo]))
        self._veloPub.publish(motor_velo)

    def get_moving_platform_pose(self):
        return (self._moving_platform_pose.pose.position.x, self._moving_platform_pose.pose.position.y,
                self._moving_platform_pose.pose.position.z,
                [self._moving_platform_pose.pose.orientation.x, self._moving_platform_pose.pose.orientation.y,
                 self._moving_platform_pose.pose.orientation.z, self._moving_platform_pose.pose.orientation.w])

    def get_cable_length(self):
        r1 = 0.04
        r2 = 0.04
        r3 = 0.04
        r4 = 0.04
        cable_length = np.array([0.0, 0.0, 0.0, 0.0])
        cable_length[0] = self._motor_pos[0] * r1 + self._ori_cable_lengths[0]
        cable_length[1] = self._motor_pos[1] * r2 + self._ori_cable_lengths[1]
        cable_length[2] = self._motor_pos[2] * r3 + self._ori_cable_lengths[2]
        cable_length[3] = self._motor_pos[3] * r4 + self._ori_cable_lengths[3]
        return cable_length

    # def pre_tension(self, cable1_flag, cable2_flag, cable3_flag, cable4_flag):
    #
    #     if cable1_flag:
    #         time.sleep(0.5)
    #         # cable1 pre-tightening
    #         print('cable1 pre-tension...')
    #         self.set_motor_velo(-50, 0, 0, 0)
    #         x0, y0, z0, _ = self.get_moving_platform_pose()
    #         while True:
    #             x, y, z, _ = self.get_moving_platform_pose()
    #             if np.linalg.norm(np.array([x, y, z]) - np.array([x0, y0, z0]), ord=2) > 0.005:
    #                 self.set_motor_velo(0, 0, 0, 0)
    #                 break
    #             else:
    #                 time.sleep(0.1)
    #
    #     if cable2_flag:
    #         time.sleep(0.5)
    #         # cable2 pre-tightening
    #         print('cable2 pre-tension...')
    #         self.set_motor_velo(0, -50, 0, 0)
    #         x0, y0, z0, _ = self.get_moving_platform_pose()
    #         while True:
    #             x, y, z, _ = self.get_moving_platform_pose()
    #             if np.linalg.norm(np.array([x, y, z]) - np.array([x0, y0, z0]), ord=2) > 0.005:
    #                 self.set_motor_velo(0, 0, 0, 0)
    #                 break
    #             else:
    #                 time.sleep(0.1)
    #
    #     if cable3_flag:
    #         time.sleep(0.5)
    #         # cable3 pre-tightening
    #         print('cable3 pre-tension...')
    #         self.set_motor_velo(0, 0, -50, 0)
    #         x0, y0, z0, _ = self.get_moving_platform_pose()
    #         while True:
    #             x, y, z, _ = self.get_moving_platform_pose()
    #             if np.linalg.norm(np.array([x, y, z]) - np.array([x0, y0, z0]), ord=2) > 0.005:
    #                 self.set_motor_velo(0, 0, 0, 0)
    #                 break
    #             else:
    #                 time.sleep(0.1)
    #
    #     if cable4_flag:
    #         time.sleep(0.5)
    #         # cable4 pre-tightening
    #         print('cable4 pre-tension...')
    #         self.set_motor_velo(0, 0, 0, -50)
    #         x0, y0, z0, _ = self.get_moving_platform_pose()
    #         while True:
    #             x, y, z, _ = self.get_moving_platform_pose()
    #             if np.linalg.norm(np.array([x, y, z]) - np.array([x0, y0, z0]), ord=2) > 0.005:
    #                 self.set_motor_velo(0, 0, 0, 0)
    #                 break
    #             else:
    #                 time.sleep(0.1)

    def loosen(self):
        print('loosening...')
        self.set_motor_velo(0.3, 0.3, 0.3, 0.3)
        time.sleep(0.2)
        self.set_motor_velo(0, 0, 0, 0)
        time.sleep(0.5)


if __name__ == "__main__":

    cdpr = CDPR()
    rate = rospy.Rate(10)
    # cdpr.pretighten()
    time.sleep(3)
    # cdpr.pretighten(True, True, True, True)
    cdpr.init_cable_length(True, True, True, True)
    time.sleep(1)
    cdpr.set_motor_velo(0, 0, 0, 0)
    start_time = time.time()
    while time.time() - start_time < 5:
        # cdpr.set_motor_velo(0, 100, 0, 0)
        # print(cdpr.get_moving_platform_pose())
        print(cdpr.get_cable_length())
        time.sleep(0.05)
    cdpr.set_motor_velo(0, 0, 0, 0)

