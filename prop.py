from utils import interp, clamp
# 初始化：发动机数据，所需发动机参数，数据的所有维度
# 输入：马赫数，高度，油门
# 输出：推力
class PropulsionCalculator:
    def __init__(self, data_prop, var_prop, grid_axes_prop):
        self.data_prop = data_prop
        self.var_prop = var_prop
        self.grid_axes_prop = grid_axes_prop

    def cal_prop(self, mach, alt, thr):
        """Calculate the thrust based on mach, altitude, and throttle"""
        # Clamp values to ensure they are within valid ranges
        mach = clamp(mach, self.data_prop['Ma_lst'].min(), self.data_prop['Ma_lst'].max())
        alt = clamp(alt, self.data_prop['alt_lst'].min(), self.data_prop['alt_lst'].max())
        thr = clamp(thr, self.data_prop['n_lst'].min(), self.data_prop['n_lst'].max())

        # Interpolate to get the desired properties
        interp_prop = {}
        for var in self.var_prop:
            interp_prop[var] = interp(self.data_prop, var, self.grid_axes_prop, method='linear')

        # Calculate thrust and coefficient of friction (if needed)
        Thrust = interp_prop['Thrust_tab'](mach, alt, thr)
        Cf = interp_prop['Cf_tab'](mach, alt, thr)

        return Thrust
