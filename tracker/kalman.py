# trackers/kalman.py
import numpy as np

class KalmanFilter2D:
    """
    Simple Kalman Filter for 2D position + velocity state:
    state vector: [x, y, vx, vy]
    """
    def __init__(self, x=0, y=0, vx=0, vy=0, dt=1.0,
                 process_var=1.0, meas_var=10.0):
        self.dt = dt
        # State vector
        self.x = np.array([x, y, vx, vy], dtype=float).reshape(4,1)

        # State transition
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)

        # Measurement matrix (we measure x,y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        # Covariances
        q = process_var
        self.Q = q * np.eye(4)  # process noise
        r = meas_var
        self.R = r * np.eye(2)  # measurement noise

        self.P = np.eye(4) * 500.0  # initial estimate covariance

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        """
        z: measurement [x, y] as iterable
        """
        z = np.array(z, dtype=float).reshape(2,1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.F.shape[0])
        self.P = (I - K @ self.H) @ self.P
        return self.x

    def get_state(self):
        return self.x.ravel()  # x,y,vx,vy
