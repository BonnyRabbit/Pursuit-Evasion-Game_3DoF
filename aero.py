from utils import interp, clamp
# 初始化：气动数据，所需气动参数，数据的所有维度,
# 输入：当前高度，真空速，迎角，侧滑角，马赫数，机翼面积
# 输出：升/阻/侧力
class AeroForceCalculator:
    
    def __init__(self, data_aero, var_aero, grid_axes_aero):
        self.data_aero = data_aero
        self.var_aero = var_aero
        self.grid_axes_aero = grid_axes_aero
    
    def _clamp_values(self):
        """Clamp the input values for alpha, beta, and mach to the valid ranges."""
        self.alpha = clamp(self.alpha, self.data_aero['aoa_lst'].min(), self.data_aero['aoa_lst'].max())
        self.beta = clamp(self.beta, self.data_aero['aos_lst'].min(), self.data_aero['aos_lst'].max())
        self.mach = clamp(self.mach, self.data_aero['ma_lst'].min(), self.data_aero['ma_lst'].max())
    
    def _cal_aero_coef(self):
        """Calculate aerodynamic coefficients based on input parameters."""
        self._clamp_values()

        # Interpolate aerodynamic coefficients based on the current conditions
        interp_aero = {var: interp(self.data_aero, var, self.grid_axes_aero, method='linear') for var in self.var_aero}

        # Get aerodynamic coefficients using interpolation
        CD = interp_aero['CD_lon'](self.alpha, self.beta, self.mach)
        CL = interp_aero['CL_lon'](self.alpha, self.beta, self.mach)
        CY = interp_aero['CY_lat'](self.alpha, self.beta, self.mach)

        return CD, CL, CY
    
    def _cal_rho(self):
        """Calculate air density based on altitude."""
        temp = 273.15  # Temperature in Kelvin
        P0 = 101325    # Sea level pressure in Pascals
        R = 287.05     # Specific gas constant for air in J/(kg·K)
        
        # Calculate pressure at the given altitude using the barometric formula
        P = P0 * (1 - 2.25577e-5 * self.alt) ** 5.2561
        
        # Calculate air density
        rho = P / (R * temp)
        return rho
    
    def cal_aero_force(self, alpha, beta, mach, tas, alt, RefArea):
        self.alpha = alpha
        self.beta = beta
        self.mach = mach
        self.tas = tas
        self.alt = alt
        self.RefArea = RefArea
        """Calculate aerodynamic forces (drag, lift, and side force)."""
        CD, CL, CY = self._cal_aero_coef()  # Calculate aerodynamic coefficients
        rho = self._cal_rho()  # Calculate air density
        
        # Calculate aerodynamic forces
        D = 0.5 * rho * self.tas**2 * self.RefArea * CD  # Drag force
        L = 0.5 * rho * self.tas**2 * self.RefArea * CL  # Lift force
        Y = 0.5 * rho * self.tas**2 * self.RefArea * CY  # Side force
        
        return D, L, Y