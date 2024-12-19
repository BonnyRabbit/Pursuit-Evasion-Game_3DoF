import scipy.io

class ModelLoader:
    def __init__(self):
        self.data_aero = None
        self.data_prop = None

    def load_mdl(self, aero_file='AeroData.mat', prop_file='PropData.mat'):
        """Load aerodynamic and propulsion model data from given files."""
        # Load aerodynamic data
        self.data_aero = scipy.io.loadmat(aero_file)
        self.data_prop = scipy.io.loadmat(prop_file)

        # Pre-process data and return
        var_aero = ['CD_lon', 'CL_lon', 'CY_lat']
        grid_axes_aero = ['aoa_lst', 'aos_lst', 'ma_lst']

        var_prop = ['Thrust_tab', 'Cf_tab']
        grid_axes_prop = ['Ma_lst', 'alt_lst', 'n_lst']
        
        return {
            'data_aero': self.data_aero,
            'data_prop': self.data_prop,
            'var_aero': var_aero,
            'grid_axes_aero': grid_axes_aero,
            'var_prop': var_prop,
            'grid_axes_prop': grid_axes_prop
        }
