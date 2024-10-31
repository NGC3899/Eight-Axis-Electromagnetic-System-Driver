import numpy as np

class PDController:
    def __init__(self, A, kp, kd, desired_P, current_P):
        self.A = A
        self.kp = kp
        self.kd = kd
        self.e_prev = 0

    def Calculation_TF(self, desired_P, current_P, dt):
        e = desired_P - current_P
        dedt = (e - self.e_prev) / dt
        pd_output = self.kp @ e + self.kd @ dedt
        self.e_prev = e
        return pd_output
    
    def Calculation_Current(self, pd_output):
        current = np.linalg.pinv(self.A) @ pd_output
        return current