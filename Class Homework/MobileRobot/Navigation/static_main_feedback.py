from vision import Vision
from action import Action
from debug import Debugger
import time

from PathPlanning import PathGenerating
import PathTracking.feedback as feedback
from PathTracking.PID import PID

if __name__ == '__main__':
    vision = Vision()
    debugger = Debugger()
    goal = [[-2400, -1500], [2400, 1500]]
    
    i = 0
    while True:
        time.sleep(0.02)
        start_x, start_y = vision.my_robot.x, vision.my_robot.y
        
        goal_x, goal_y = goal[i%2][0], goal[i%2][1]
        
        path_x, path_y, path_theta = PathGenerating.path_generating(vision, start_x, start_y, goal_x, goal_y)
        
        debugger.draw_path(path_x, path_y)
        path_x.pop(0)
        path_y.pop(0)    
        
        feedback.move_along_path(vision, path_x, path_y, path_theta)
        i += 1
        
        # pid_kwargs = {
        #     'path_x': path_x,
        #     'path_y': path_y,
        #     'path_theta': path_theta,
        #     'kp': 5 * np.ones(3),
        #     'ki': 20 * np.ones(3),
        #     'kd': np.zeros(3),
        #     'dt': 0.05
        # }
        
        # controller = PID(**pid_kwargs)
        # controller.control(vision)
        
            
            
            
            