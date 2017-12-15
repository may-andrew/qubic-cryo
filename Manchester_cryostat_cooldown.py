###################

# Cryostat cooldown thermal simulation - Andy May, Oct 2017
# andrew.may-3@postgrad.manchester.ac.uk

###################

# The following code models the cooldown of a medium-sized 2 stage cryostat by a PT415 pulse tube cryostat

# The capacity map of the mechanical coolers are used from Green et al. (2015)
# http://iopscience.iop.org/article/10.1088/1757-899X/101/1/012002/pdf

# Radiative loads are accounted for using typical shield sizings with emissivity consistent with MLI

# The thermal masses on the 1st and 2nd stages are modelled as lumped elements (i.e. isothermal)
# These stages are treated as copper with temperature dependent heat capacity from Marquardt, Le, and Radebaugh
# Cryogenic Material Properties Database, National Institute of Standards and Technology Boulder, CO 80303

###################

# Import packages

import numpy as np
import scipy as sp
import matplotlib
from matplotlib import pyplot as plt
from scipy import interpolate

###################

# Model time

endtime = 5 # hours
endtime = endtime * 3600 # secs
timestep = 0.1 # secs
globaltime = np.array([0.00]) # secs # initialise, will increase by "timestep" until reaches "endtime"

###################

# Define physical constants

stefboltz = 5.67E-8 # W/m2.K4

###################

# Define initial conditions and parameters for thermal elements

# Element 0 # Outer vacuum chamber, model as thermal reservoir at 300 K

T_0 = np.array([300.0]) # K # Initialise temperature array

# Element 1 # 1st stage, thermal mass

A_1 = 0.5 # m2 # Surface area
eps1 = 0.01 # Surface emissivity
m_1 = 12 # kg # Mass
T_1 = np.array([300.0]) # K # Initialise temperature array

# Element 2 # 2nd stage, thermal mass

A_2 = 0.5 # m2 # Surface area
eps2 = 0.01 # Surface emissivity
m_2 = 15 # kg # Mass
T_2 = np.array([300.0]) # K # Initialise temperature array

###################

# Initialise heat flow arrays

Qdot1_record = np.array([0.00]) # W # PTC 1st stage heat lift
Qdot2_record = np.array([0.00]) # W # PTC 2nd stage heat lift

QRdot1_record = np.array([0.00]) # W # Radiative load on 1st stage
QRdot2_record = np.array([0.00]) # W # Radiative load on 2nd stage

###################

# PTC heat lift data

T_PTC1 = np.array([31.42, 42.26, 41.72, 41.72, 41.72, 42.26, 49.85, 57.43, 59.06, 59.06, 60.68, 62.31, 75.31, 85.06, 89.40, 100.23, 115.94, 122.99, 114.32,141.41, 153.33, 154.95, 163.08, 175.54, 216.18, 218.89, 236.76, 246.52, 250.85, 286.61, 295.82, 303.95, 308.28]) # K # PTC 1st stage temperature
T_PTC2 = np.array([2.51, 33.52, 76.26, 144.97, 248.04, 292.46, 2.52, 36.03, 78.77, 144.97, 224.58, 310.06, 3.35, 40.22, 79.61, 134.08, 201.12, 282.40, 5.03, 51.12, 97.21, 151.68, 222.07, 315.08, 41.07, 82.96, 150.00, 232.122, 320.11, 70.39, 130.72, 204.47, 284.92]) # K # PTC 2nd stage temperature
Q_PTC1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 200.0, 200.0, 200.0, 200.0, 200.0, 250.0, 250.0, 250.0, 250.0]) # W # PTC 1st stage heat lift
Q_PTC2 = np.array([0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 0.0, 25.0, 50.0, 75.0, 100.0, 0.0, 25.0, 50.0, 75.0]) # W # PTC 2nd stage heat lift

# Interpolated values as f(T_1,T_2)

Q_PTC1_interp_func = interpolate.interp2d(T_PTC1, T_PTC2, Q_PTC1, kind='linear') # W # PTC 1st stage heat lift
Q_PTC2_interp_func = interpolate.interp2d(T_PTC1, T_PTC2, Q_PTC2, kind='linear') # W # PTC 2nd stage heat lift

###################

# Solver

while globaltime[-1] < endtime : # Loop until simulation reaches endtime
    
    ###################
    
    # Calculate temperature dependent properties
    
    # Element 0
    
    # 
    
    # Element 1
    
    cp_1 = 10**((-1.91844) + (-0.15973*np.log10(T_1[-1])) + (8.61013*(np.log10(T_1[-1])**2)) + (-18.99640*(np.log10(T_1[-1])**3)) + (21.96610*(np.log10(T_1[-1])**4)) + (-12.73280*(np.log10(T_1[-1])**5) + (3.54322*(np.log10(T_1[-1])**6))) + (-0.37970*(np.log10(T_1[-1])**7)))
    Cp_1 = cp_1 * m_1 # J/K
    
    # Element 2
    
    cp_2 = 10**((-1.91844) + (-0.15973*np.log10(T_2[-1])) + (8.61013*(np.log10(T_2[-1])**2)) + (-18.99640*(np.log10(T_2[-1])**3)) + (21.96610*(np.log10(T_2[-1])**4)) + (-12.73280*(np.log10(T_2[-1])**5) + (3.54322*(np.log10(T_2[-1])**6))) + (-0.37970*(np.log10(T_2[-1])**7)))
    Cp_2 = cp_2 * m_2 # J/K
    
    ###################
    
    # Calculate heat transfer
    
    # Element 0
    
    # 
    
    # Element 1
    
    # PTC heat lift
    
    Qdot1 = Q_PTC1_interp_func(T_1[-1], T_2[-1]) # W # 1st stage heat lift as function of 1st and 2nd stage temperatures
    Q1 = Qdot1 * timestep # J # Heat removed in timestep
    Qdot1_record = np.append(Qdot1_record,Qdot1) # W # Record for plotting
    
    # Radiative load 
    
    QRdot1 = A_1 * eps1 * stefboltz * ((T_0[-1] **4) - (T_1[-1] **4)) # W # Rad load on 1st stage
    QR1 = QRdot1 * timestep # J # Radiative heat added in timestep
    QRdot1_record = np.append(QRdot1_record,QRdot1) # W # Record for plotting
    
    # Element 2
    
    # PTC heat lift
    
    Qdot2 = Q_PTC2_interp_func(T_1[-1], T_2[-1]) # W # 2nd stage heat lift as function of 1st and 2nd stage temperatures
    Q2 = Qdot2 * timestep # J # Heat removed in timestep
    Qdot2_record = np.append(Qdot2_record,Qdot2) # W # Record for plotting
    
    # Radiative load
    
    QRdot2 = A_2 * eps2 * stefboltz * ((T_1[-1] **4) - (T_2[-1] **4)) # W # Rad load on 2nd stage
    QR2 = QRdot2 * timestep # J # Radiative heat added in timestep
    QRdot2_record = np.append(QRdot2_record,QRdot2) # W # Record for plotting
    
    ###################
    
    # Calculate temperatures at end of timestep
    
    # Element 0
        
    T_0_new = T_0[-1] # K # Thermal reservoir, hence temperature unchanged
    T_0 = np.append(T_0,T_0_new) # Add new temperature to array
    
    
    # Element 1
        
    T_1_new = ( T_1[-1] ) + ( ( -Q1 + QR1 -QR2 ) / Cp_1) # K # Temperature change from net heat transfer divided by total heat capacity
    T_1 = np.append(T_1,T_1_new) # K # Add new temperature to array
    
    # Element 2

    T_2_new = ( T_2[-1] ) + ((-Q2+QR1) / Cp_2) # K # Temperature change from net heat transfer divided by total heat capacity
    T_2 = np.append(T_2,T_2_new) # K # Add new temperature to array
    
    ###################
    
    # Timekeeping
    
    time_new = globaltime[-1] + timestep # Move time on by one timestep 
    globaltime = np.append(globaltime,time_new) # Add new time to array
    
    ###################

###################

# Plotting

# Plot temperatures over time

plt.figure() # Initialise figure

globaltime = globaltime / 3600 # hours

plt.plot(globaltime, T_0, label='Vacuum chamber') # Plot vacuum chamber temperature
plt.plot(globaltime, T_1, label='1st stage') # Plot 1st stage temperature
plt.plot(globaltime, T_2, label='2nd stage') # Plot 2nd stage temperature

plt.xlabel('Time [hours]') # x-axis label
plt.ylabel('Temperature [K]') # y-axis label
plt.title('Thermal simulation of Manchester cryostat cooldown') # Title
plt.legend(loc=1) # Legend position

x1,x2,y1,y2 = plt.axis() # Define axis ranges
plt.axis((x1,x2,y1,310)) # Define axis ranges

# Plot heat transfer rates

plt.figure() # Initialise figure

plt.plot(globaltime, Qdot1_record, label='PTC1 heat lift') # Plot PTC 1st stage heat lift
plt.plot(globaltime, Qdot2_record, label='PTC2 heat lift') # Plot PTC 2nd stage heat lift
plt.plot(globaltime, QRdot1_record, label='Radiative load on 1st stage') # Plot 1st stage radiative heat load
plt.plot(globaltime, QRdot2_record, label='Radiative load on 2nd stage') # Plot 2nd stage radiative heat load

plt.xlabel('Time [hours]') # x-axis label
plt.ylabel('Qdot [W]') # y-axis label
plt.title('Thermal simulation of Manchester cryostat cooldown') # Title
plt.legend(loc=1) # Legend position

x1,x2,y1,y2 = plt.axis() # Define axis ranges
plt.axis((0,x2,0,y2)) # Define axis ranges

plt.show() # Display plots

###################

# End

###################