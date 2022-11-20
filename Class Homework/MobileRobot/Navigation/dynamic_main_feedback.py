from random import sample
from vision import Vision
from action import Action
from debug import Debugger
import time

import PathTracking.feedback as feedback

if __name__ == '__main__':
    vision = Vision()
    debugger = Debugger()
    goal = [[-2400, -1500], [2400, 1500]]
    i = 0
    
    while True:
        time.sleep(0.02)
        goal_x, goal_y = goal[i%2][0], goal[i%2][1]
        feedback.move_to_goal(vision, goal_x, goal_y)
        i += 1
        
        
        
            
            
            
            