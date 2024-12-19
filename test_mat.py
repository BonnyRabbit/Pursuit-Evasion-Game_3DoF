import scipy.io

aero_data = scipy.io.loadmat('AeroData.mat')
# mass_data = scipy.io.loadmat('MassBalanceData.mat')
# prop_data = scipy.io.loadmat('PropData1102.mat')

aero_data = aero_data['aero_data']