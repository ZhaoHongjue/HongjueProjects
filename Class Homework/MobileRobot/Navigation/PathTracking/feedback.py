from action import Action
from debug import Debugger
import math
import time
import numpy as np
from PathPlanning.PathGenerating import path_generating
          
def move_to_pose(vision, path_x, path_y):
    Kp_rho = 3000
    Kp_alpha = 15
    Kp_beta = -3
    dt = 0.01
    
    action = Action()
    sth_lst = [math.pi]
    gth_lst = []
    
    for i in range(1, len(path_x)):
        x_goal, y_goal = path_x[i] / 1000, path_y[i] / 1000
        x_start, y_start = path_x[i-1] / 1000, path_y[i-1] / 1000
        # print(f'sx: {x_start} sy: {y_start} gx: {x_goal} gy: {y_goal}')
        gth = math.atan2((y_goal - y_start), (x_goal - x_start))
        gth_lst.append(gth)
        x, y = x_start, y_start
        theta = sth_lst[-1]

        x_diff = x_goal - x
        y_diff = y_goal - y

        x_traj, y_traj = [], []

        rho = np.hypot(x_diff, y_diff)
        while rho > 0.1:
            x_traj.append(x)
            y_traj.append(y)

            x_diff = x_goal - x
            y_diff = y_goal - y

            rho = np.hypot(x_diff, y_diff)
            alpha = (np.arctan2(y_diff, x_diff)
                 - theta + np.pi) % (2 * np.pi) - np.pi
            beta = (gth_lst[-1] - theta - alpha + np.pi) % (2 * np.pi) - np.pi

            v = Kp_rho * rho
            w = Kp_alpha * alpha + Kp_beta * beta

            if alpha > np.pi / 2 or alpha < -np.pi / 2:
                v = -v
        
            # print(f'v:{v} w:{w}')
            action.sendCommand(vx = v, vw = w)
            time.sleep(dt)

            theta = vision.my_robot.orientation
            x = vision.my_robot.x / 1000
            y = vision.my_robot.y / 1000
            sth_lst.append(gth)
        
def move_along_path(vision, path_x, path_y, path_theta):
    
    Kp_rho = 3000
    Kp_alpha = 15
    Kp_beta = -3
    dt = 0.01
    
    path_x = np.asarray(path_x, dtype = float) / 1000
    path_y = np.asarray(path_y, dtype = float) / 1000
    path_theta = np.asarray(path_theta, dtype = float)
    action = Action()
    
    while True:
        x = vision.my_robot.x / 1000
        y = vision.my_robot.y / 1000
        theta = vision.my_robot.orientation
        if math.hypot(x - path_x[-1], y - path_y[-1]) < 0.1:
            break
        i = np.argmin(np.hypot(path_x - x, path_y - y))
        ind = min(i + 50, len(path_x) - 1)
        gx, gy = path_x[ind], path_y[ind]
        gth = path_theta[ind]
        
        x_diff = gx - x
        y_diff = gy - y
        
        rho = np.hypot(x_diff, y_diff)
        alpha = (np.arctan2(y_diff, x_diff)
                 - theta + np.pi) % (2 * np.pi) - np.pi
        beta = (gth - theta - alpha + np.pi) % (2 * np.pi) - np.pi

        v = Kp_rho * rho
        w = Kp_alpha * alpha + Kp_beta * beta

        if alpha > np.pi / 2 or alpha < -np.pi / 2:
            v = -v
        
        action.sendCommand(vx = v, vw = w)
        time.sleep(dt)
        
def move_to_goal(vision, goal_x, goal_y):
    action = Action()
    debugger = Debugger()
    Kp_rho = 350
    Kp_alpha = 2
    Kp_beta = -0.5
    dt = 0.01
    
    while True:
        # time.sleep(0.1)
        x = vision.my_robot.x
        y = vision.my_robot.y
        theta = vision.my_robot.orientation
        path_x, path_y, path_theta = path_generating(vision, x, y, goal_x, goal_y)
        
        if len(path_x) == 0:
            continue
        
        debugger.draw_path(path_x, path_y)
        
        path_x.pop(0)
        path_y.pop(0)
        path_x = np.asarray(path_x, dtype = float) / 1000
        path_y = np.asarray(path_y, dtype = float) / 1000
        path_theta = np.asarray(path_theta, dtype = float)
        
        x /= 1000
        y /= 1000
        
        # print(math.hypot(x - path_x[-1], y - path_y[-1]))
        if math.hypot(x - path_x[-1], y - path_y[-1]) < 0.1:
            # print('OK')
            break
        i = np.argmin(np.hypot(path_x - x, path_y - y))
        ind = min(i + 100, len(path_x) - 1)
        gx, gy = path_x[ind], path_y[ind]
        gth = path_theta[ind]
        
        x_diff = gx - x
        y_diff = gy - y
        
        rho = np.hypot(x_diff, y_diff)
        alpha = (np.arctan2(y_diff, x_diff)
                 - theta + np.pi) % (2 * np.pi) - np.pi
        beta = (gth - theta - alpha + np.pi) % (2 * np.pi) - np.pi

        v = Kp_rho * rho
        w = Kp_alpha * alpha + Kp_beta * beta

        if alpha > np.pi / 2 or alpha < -np.pi / 2:
            v = -v
        
        action.sendCommand(vx = v, vw = w)
        time.sleep(dt)