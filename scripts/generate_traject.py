import numpy as np
import matplotlib.pyplot as plt


def smooth_p2p(way_points_list, travel_time_list, velo_limit, time_step):
    traject = []

    for index, travel_time in enumerate(travel_time_list):
        max_velo = (way_points_list[index+1] - way_points_list[index]) / (0.75 * travel_time)
        if np.linalg.norm(max_velo) > velo_limit:
            return None
        else:
            step_number = travel_time / time_step
            pos = way_points_list[index].copy()     # 真坑啊 nparray
            for step in range(int(step_number)):
                if step < 0.25 * step_number:
                    velo = max_velo * step / (0.25 * step_number)
                elif step < 0.75 * step_number:
                    velo = max_velo
                else:
                    velo = max_velo * (1 - (step - 0.75 * step_number) / (0.25 * step_number))
                pos += velo * time_step
                traject.append(pos.copy())

    return traject


if __name__ == '__main__':

    zero_pos = np.array([0, 0, 0.430])
    target1 = np.array([0.445, 0.210, 0.078]) - np.array([0.249, 0.1515, 0])
    safe_point1 = np.array([0.200, 0.150, 0.250])  # 安全位置1
    safe_point2 = np.array([-0.200, -0.150, 0.250])  # 安全位置2
    safe_point3 = np.array([-0.200, 0.150, 0.250])
    traject_height = -0.050 + 0.172  # 轨迹高度（实测）
    traject_1 = np.array([0.153, 0.000, traject_height])
    traject_2 = np.array([0.153, -0.153, traject_height])
    traject_3 = np.array([-0.160, -0.153, traject_height])
    traject_4 = np.array([-0.160, 0.153, traject_height])

    waypoints = [zero_pos, safe_point1, target1 + np.array([0, 0, 0.050]), target1, target1, traject_1,
                 traject_2, traject_3, traject_4, traject_4, safe_point3, zero_pos]
    travel_time = [10, 8, 6, 6, 10, 15, 30, 30, 6, 8, 10]
    velo_limit = 0.100
    time_step = 0.1

    traject = smooth_p2p(waypoints, travel_time, velo_limit, time_step)
    traject = np.array(traject)

    np.savetxt("trajectory+.txt", traject)

    # plot
    fig = plt.figure(1)
    x_plot = fig.add_subplot(4, 2, 1)
    y_plot = fig.add_subplot(4, 2, 3)
    z_plot = fig.add_subplot(4, 2, 5)

    x_plot.plot(traject[:, 0])
    y_plot.plot(traject[:, 1])
    z_plot.plot(traject[:, 2])

    plt.ioff()
    plt.show()

