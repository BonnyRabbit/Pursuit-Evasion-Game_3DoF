import numpy as np
import math
from load_mdl import ModelLoader
from utils import scale, DCM
from aero import AeroForceCalculator
from prop import PropulsionCalculator

class fdm_3DoF():
    
    def __init__(self):

        # Iniialize model
        self.model_loader = ModelLoader()
        model_data = self.model_loader.load_mdl()
        # Pre-process data
        self.data_aero = model_data['data_aero']
        self.data_prop = model_data['data_prop']
        self.var_aero = model_data['var_aero']
        self.grid_axes_aero = model_data['grid_axes_aero']
        self.var_prop = model_data['var_prop']
        self.grid_axes_prop = model_data['grid_axes_prop']

        """
        Params
        """
        self.dt_AP = 1
        self.mass = 200
        self.g = 9.8
        self.RefArea = self.data_aero['RefArea']
        self.RefLen = self.data_aero['RefLen']
        self.RefSpan = self.data_aero['RefSpan']
        """
        State
        """
        # [x, y, z, x_t, y_t, z_t, v, alpha, beta, gamma, chi, chi_t, mu]
        # speed
        self.v = None
        self.mach = None
        # position
        self.x = None
        self.y = None
        self.z = None
        self.x_t = None
        self.y_t = None
        self.z_t = None
        self.alt = None
        # attitude angle
        self.phi = None
        self.tht = None
        self.psi = None
        # aero angle
        self.alpha = None
        self.beta = None
        # track angle
        self.gamma = None
        self.chi = None
        self.chi_t = None
        self.mu = None
        """
        Action
        """
        # [ay, az, thr]
        self.action_type = ['alpha', 'beta', 'thr']
        self.a_y = None
        self.a_z = None
        self.thr = None
        """
        Saturation
        """

        # Inititalize aero_cal
        self.aero_force = AeroForceCalculator(
            data_aero=self.data_aero,
            var_aero=self.var_aero,
            grid_axes_aero=self.grid_axes_aero,
        )
        # Initialize prop_cal
        self.prop_force = PropulsionCalculator(
            data_prop=self.data_prop,
            var_prop=self.var_prop,
            grid_axes_prop=self.grid_axes_prop
        )     
        
 
    def cal_aero_force(self):
        D, L, Y = self.aero_force.cal_aero_force(
            alpha=self.alpha,
            beta=self.beta,
            mach=self.mach,
            v=self.v,
            alt=self.alt,
            RefArea=self.RefArea
        )
        D = D.item() if isinstance(D, np.ndarray) else D
        L = L.item() if isinstance(L, np.ndarray) else L
        Y = Y.item() if isinstance(Y, np.ndarray) else Y

        return D, L, Y

    def cal_prop_force(self):
        Thrust = self.prop_force.cal_prop(
            mach=self.mach,
            alt=self.alt,
            thr=self.thr
        )
        Thrust = Thrust.item() if isinstance(Thrust, np.ndarray) else Thrust
        return Thrust

    def cal_mass(self):
        pass
    
    
    def cal_mach(self):
        temp= 273.15
        gamma = 1.4
        R = 287.05
        T = temp - 0.0065 * self.alt
        c = math.sqrt(gamma * R * T)
        self.mach = self.v / c

        return self.mach

    def DCM_kg(self):
        Ly_gamma = DCM('y',self.gamma)
        Lz_chi = DCM('z', self.chi)
        L_kg = Ly_gamma @ Lz_chi

        return L_kg
    
    def DCM_ka(self):
        Lx_mu = DCM('x', self.mu)
        L_ka = np.linalg.inv(Lx_mu)

        return L_ka
    
    def DCM_ab(self):
        Ly_alpha = DCM('y', -self.alpha)
        Lz_beta = DCM('z', self.beta)
        L_ab = Lz_beta @ Ly_alpha

        return L_ab
    
    def take_action(self, action):
        # Update state
        alpha = self.alpha
        beta = self.beta
        mach = self.cal_mach()
        v = self.v
        alt = self.alt
        thr = self.thr
        mass = self.mass
        x = self.x
        y = self.y
        z = self.z
        gamma = self.gamma
        mu = self.mu
        chi = self.chi

        # Actions scale
        delta_alpha, delta_beta, delta_thr = [scale(i, type) for i, type in zip(action, self.action_type)]


        self.alpha = alpha + math.radians(delta_alpha) * self.dt_AP
        self.beta = beta + math.radians(delta_beta) * self.dt_AP
        self.thr = thr + delta_thr * self.dt_AP

        D, L, Y = self.cal_aero_force()
        T = self.cal_prop_force()
        
        """
        6th Order Point Mass Flight Model in Kinetic axis
        m * (Vx_dot + Vz*wy - Vy*wz) = Fx
        m * (Vy_dot + Vx*wz - Vz*wx) = Fy
        m * (Vz_dot + Vy*wx - Vx*wy) = Fz
        """
        L_kg = self.DCM_kg()
        L_ka = self.DCM_ka()
        L_ab = self.DCM_ab()
        L_gk = np.linalg.inv(L_kg)
        L_kb = L_ka @ L_ab
        v_g = L_gk @ np.array([[v], [0], [0]])
        dx, dy, dz = v_g.flatten()
        F = (L_kb @ np.array([[T], [0], [0]])
                        + L_ka @ np.array([[-D], [Y], [-L]])
                        + L_kg @ np.array([[0], [0], [mass * self.g]]))
        omega = (L_kg @ np.array([[0], [0], [chi]]) + np.array([[0], [gamma], [0]])).flatten()
        v_dot, gamma_dot, chi_dot = F / (mass * omega)
        
        self.x = x + dx * self.dt_AP
        self.y = y + dy * self.dt_AP
        self.z = z + dz * self.dt_AP
        self.v = v + v_dot * self.dt_AP
        self.gamma = gamma + gamma_dot * self.dt_AP
        self.chi = chi + chi_dot * self.dt_AP


#     # Example usage
# fdm = fdm_3Dof()
# fdm.mach = 0.8
# fdm.alt = 1000  # meters
# fdm.thr = 0.75  # throttle setting

# Thrust = fdm.cal_prop_force()
# print(f"Thrust: {Thrust}")

# fdm = fdm_3Dof()
# fdm.alpha = 5.0
# fdm.beta = 0
# fdm.mach = 0.8
# fdm.v = 250  # m/s
# fdm.alt = 1000  # meters

# D, L, Y = fdm.cal_aero_force()
# print(f"Drag: {D}, Lift: {L}, Side Force: {Y}")