import math
from sys import path
from vision import Vision
from action import Action
from debug import Debugger
import numpy as np

from PathPlanning.Voronoi import VoronoiMap
from PathPlanning.prm import PRM
from PathPlanning.PotentialField import APF

def path_generating(vision, start_x, start_y, goal_x, goal_y):
    planner1 = VoronoiMap()
    planner2 = APF()
    
    path_x, path_y = planner1.plan(vision, start_x, start_y, goal_x, goal_y)
    path_x_final, path_y_final = [], []
        
    for i in range(1, len(path_x)):
        x_start, y_start = path_x[i-1], path_y[i-1]
        x_goal, y_goal = path_x[i], path_y[i]
        # print(f'sx: {x_start} sy: {y_start} gx: {x_goal} gy: {y_goal}')
        try:
            road_points_x, road_points_y = planner2.potential_field_path_planning(vision, x_start, y_start, x_goal, y_goal)
        except:
            road_points_x = np.linspace(x_start, x_goal, 100).tolist()
            road_points_y = np.linspace(y_start, y_goal, 100).tolist()
        # print(road_points_x)
        path_x_final += road_points_x
        path_y_final += road_points_y
    
    path_theta = []
    for i in range(1, len(path_x_final)):
        dy = path_y_final[i] - path_y_final[i-1]
        dx = path_x_final[i] - path_x_final[i-1]
        theta0 = math.atan2(dy, dx) + 2*math.pi
        path_theta.append(theta0)
        
    return path_x_final, path_y_final, path_theta