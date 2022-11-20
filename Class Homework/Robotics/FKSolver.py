import numpy as np
from math import cos, sin, atan2, sqrt, pi

def Single_T(DH: list):
    '''求解单步的T'''
    a, alpha, d, theta = DH
    return np.array([[cos(theta), -sin(theta), 0, a],
                     [sin(theta)*cos(alpha), cos(theta)*cos(alpha), -sin(alpha), -d*sin(alpha)],
                     [sin(theta)*sin(alpha), cos(theta)*sin(alpha), cos(alpha), d*cos(alpha)],
                     [0, 0, 0, 1]])

    

def FKSolver(theta: tuple):
    assert len(theta) == 6
    DH = np.array([[0,      0,      0.23,   theta[0]        ],
                   [0,      -pi/2,  -0.054, -pi/2 + theta[1]],
                   [0.185,  0,      0,      theta[2]        ],
                   [0.170,  0,      0.077,  pi/2 + theta[3] ],
                   [0,      pi/2,   0.077,  pi/2 + theta[4] ],
                   [0,      pi/2,   0,      theta[5]        ],
                   [0,      0,      0.0855, 0               ]
                ], dtype = np.float32)

    T = np.eye(4)
    for i in range(DH.shape[0]):
        ST = Single_T(DH[i].tolist())
        T = T @ ST
        
    print(T)
    
    Position = T[0:3, -1]
    
    Fixed_beta = atan2(-T[2, 0], sqrt(T[0, 0]**2 + T[1, 0]**2)) 
    Fixed_alpha = atan2(T[1, 0]/cos(Fixed_beta), T[0, 0]/cos(Fixed_beta))
    Fixed_gamma = atan2(T[2, 1]/cos(Fixed_beta), T[2, 2]/cos(Fixed_beta))
    Fixed_XYZ = (Fixed_gamma*180/pi,  Fixed_beta*180/pi, Fixed_alpha*180/pi)
    
    Euler_beta = atan2(T[0, 2], sqrt(T[0, 0]**2 + T[0, 1]**2))
    Euler_alpha = atan2(-T[1, 2]/cos(Euler_beta), T[2, 2]/cos(Euler_beta))
    Euler_gamma = atan2(-T[0, 1]/cos(Euler_beta), T[0, 0]/cos(Euler_beta))
    Euler_XYZ = (Euler_alpha*180/pi, Euler_beta*180/pi, Euler_gamma*180/pi)
    
    return Position, Euler_XYZ, Fixed_XYZ

if __name__ == '__main__':
    theta = [(pi/6,     0,      pi/6,       0,      pi/3,       0   ),
             (pi/6,     pi/6,   pi/3,       0,      pi/3,       pi/6),
             (pi/2,     0,      pi/2,       -pi/3,  pi/3,       pi/6),
             (-pi/6,    -pi/6,  -pi/3,      0,      pi/12,      pi/2),
             (pi/12,    pi/12,  pi/12,      pi/12,  pi/12,      pi/12)]
    for i in range(1):
        Position, Euler_XYZ, Fixed_XYZ = FKSolver(theta[i])
        print('-------------------------------')
        print('{}:'.format(i+1))
        print('Position:  ({:.4f}, {:.4f}, {:.4f})'.format(Position[0], Position[1], Position[2]))
        print('Euler_XYZ: ({:.4f}, {:.4f}, {:.4f})'.format(Euler_XYZ[0], Euler_XYZ[1], Euler_XYZ[2]))
        print('Fixed_XYZ: ({:.4f}, {:.4f}, {:.4f})'.format(Fixed_XYZ[0], Fixed_XYZ[1], Fixed_XYZ[2]))