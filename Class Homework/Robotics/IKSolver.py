import numpy as np
from math import cos, sin, sqrt, atan2, acos, pi

c = cos
s = sin

def Build_Transform_Rotation(Euler_XYZ: tuple):
    assert len(Euler_XYZ) == 3
    alpha, beta, gamma = Euler_XYZ

    R00 = c(beta) * c(gamma)
    R01 = -c(beta) * s(gamma)
    R02 = s(beta)
    
    R10 = s(alpha) * s(beta) * c(gamma) + c(alpha) * s(gamma)
    R11 = -s(alpha) * s(beta) * s(gamma) + c(alpha) * c(gamma)
    R12 = -s(alpha) * c(beta)
    
    R20 = -c(alpha) * s(beta) * c(gamma) + s(alpha) * s(gamma)
    R21 = c(alpha) * s(beta) * s(gamma) + s(alpha) * c(gamma)
    R22 = c(alpha) * c(beta)
    
    return np.array(
        [
            [R00,   R01,    R02],
            [R10,   R11,    R12],
            [R20,   R21,    R22],
        ]
    )
    
def IKSolver(pose, rad = True):
    theta = []
    x, y, z = pose[0:3]
    Euler_XYZ = pose[3:]
    # 求取旋转矩阵
    R = Build_Transform_Rotation(Euler_XYZ)
    
    # 求theta1(有两解)
    px = x - 0.0855 * R[0, 2]
    py = y - 0.0855 * R[1, 2]
    rho = sqrt(px**2 + py**2)
    theta11 = atan2(py, px) - atan2(0.023, sqrt(rho**2 - 0.023**2))
    theta12 = atan2(py, px) - atan2(0.023, -sqrt(rho**2 - 0.023**2))
    
    for theta1 in [theta11, theta12]:
        LS21 = -R[0, 0] * s(theta1) + R[1, 0] * c(theta1)
        LS22 = -R[0, 1] * s(theta1) + R[1, 1] * c(theta1)
        LS23 = -R[0, 2] * s(theta1) + R[1, 2] * c(theta1)
        LS33 = R[2, 2]
        LS13 = R[0, 2] * c(theta1) + R[1, 2] * s(theta1)
        
        theta51 = atan2(LS23, sqrt(LS21**2 + LS22**2))
        theta52 = atan2(LS23, -sqrt(LS21**2 + LS22**2))
        for theta5 in [theta51, theta52]:
            theta6 = atan2(-LS22/c(theta5), LS21/c(theta5))
            theta234 = atan2(-LS33/c(theta5), LS13/c(theta5))

            LS14 = px * c(theta1) + py * s(theta1)
            LS34 = z - 0.0855*R[2, 2] - 0.23
            a, b = LS14 - 0.077 * s(theta234), LS34 - 0.077 * c(theta234)
            
            c3 = (a**2 + b**2 - 0.185**2 - 0.17**2) / (2 * 0.185 * 0.17)
            if c3 > 1 or c3 < -1:
                continue
            theta31 = acos(c3)
            theta32 = -theta31
            
            for theta3 in [theta31, theta32]:
                p = 0.185 + 0.17 * c(theta3)
                q = 0.17 * s(theta3)
                
                theta2 = atan2((a*p - b*q)/(p**2 + q**2), (a*q + b*p)/(p**2 + q**2))
                theta4 = theta234 - theta2 - theta3
                theta.append([theta1, theta2, theta3, theta4, theta5, theta6])
    
    return np.asarray(theta)

if __name__ == '__main__':
    pose = (0.0905, 0.1643, 0.6075, -104.5025*pi/180, -3.3257*pi/180, -154.2947*pi/180)
    print(IKSolver(pose))
    # print(sin(-2.24980477e+00))
    # print(cos(-2.24980477e+00))