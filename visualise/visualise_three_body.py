import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import os

def learned_three_body_odes(t, x):
    # x[0:3] - body 1 (x,y,z)
    # x[3:6] - body 2 (x,y,z)
    # x[6:9] - body 3 (x,y,z)
    
    dx0 = (0.0000 + 0.0000*x[0] + 0.0000*x[1] + -0.0722*x[2] + 0.9771*x[3] + 0.0000*x[4] 
          + 0.1433*x[5] + -0.0000*x[6] + -0.0000*x[7] + 0.0000*x[8] + -0.0000*x[0]*x[0] 
          + 0.0000*x[0]*x[1] + -0.0000*x[0]*x[2] + -0.0000*x[0]*x[3] + 0.0000*x[0]*x[4] 
          + -0.0000*x[0]*x[5] + -0.0000*x[0]*x[6] + 0.0000*x[0]*x[7] + -0.0000*x[0]*x[8] 
          + 0.0000*x[1]*x[1] + 0.0000*x[1]*x[2] + -0.0000*x[1]*x[3] + 0.0000*x[1]*x[4] 
          + -0.0168*x[1]*x[5] + 0.0000*x[1]*x[6] + -0.0000*x[1]*x[7] + -0.0000*x[1]*x[8] 
          + -0.0000*x[2]*x[2] + -0.0000*x[2]*x[3] + -0.0000*x[2]*x[4] + -0.0000*x[2]*x[5] 
          + -0.0000*x[2]*x[6] + 0.0000*x[2]*x[7] + 0.3662*x[2]*x[8] + 0.0000*x[3]*x[3] 
          + 0.0000*x[3]*x[4] + -0.0000*x[3]*x[5] + -0.0000*x[3]*x[6] + -0.0000*x[3]*x[7] 
          + 0.0000*x[3]*x[8] + 0.0000*x[4]*x[4] + -0.0000*x[4]*x[5] + 0.0000*x[4]*x[6] 
          + 0.0000*x[4]*x[7] + 0.0000*x[4]*x[8] + -0.2999*x[5]*x[5] + 0.0000*x[5]*x[6] 
          + 0.1070*x[5]*x[7] + -0.1704*x[5]*x[8] + 0.0000*x[6]*x[6] + 0.0000*x[6]*x[7] 
          + -0.0000*x[6]*x[8] + -0.0000*x[7]*x[7] + 0.0000*x[7]*x[8] + -0.0000*x[8]*x[8])
    
    dx1 = (-0.0000 + 0.0452*x[0] + 0.0000*x[1] + 0.0000*x[2] + 0.0000*x[3] + 0.9439*x[4] 
          + 0.2397*x[5] + -0.0000*x[6] + 0.0000*x[7] + -0.3597*x[8] + -0.0000*x[0]*x[0] 
          + -0.0000*x[0]*x[1] + -0.1845*x[0]*x[2] + 0.0000*x[0]*x[3] + -0.0000*x[0]*x[4] 
          + -0.0000*x[0]*x[5] + -0.0000*x[0]*x[6] + -0.0581*x[0]*x[7] + -0.0000*x[0]*x[8] 
          + -0.0000*x[1]*x[1] + -0.3819*x[1]*x[2] + 0.0000*x[1]*x[3] + -0.0000*x[1]*x[4] 
          + 0.0000*x[1]*x[5] + 0.0002*x[1]*x[6] + -0.0000*x[1]*x[7] + -0.0594*x[1]*x[8] 
          + -0.0000*x[2]*x[2] + 0.1327*x[2]*x[3] + -0.0000*x[2]*x[4] + -0.0000*x[2]*x[5] 
          + -0.0000*x[2]*x[6] + 0.0000*x[2]*x[7] + 0.8913*x[2]*x[8] + -0.0000*x[3]*x[3] 
          + -0.0000*x[3]*x[4] + -0.0000*x[3]*x[5] + 0.0000*x[3]*x[6] + 0.0000*x[3]*x[7] 
          + -0.0076*x[3]*x[8] + -0.0000*x[4]*x[4] + 0.0000*x[4]*x[5] + -0.0000*x[4]*x[6] 
          + 0.0588*x[4]*x[7] + 0.0833*x[4]*x[8] + -0.6073*x[5]*x[5] + 0.1491*x[5]*x[6] 
          + 0.2729*x[5]*x[7] + 0.4500*x[5]*x[8] + -0.0000*x[6]*x[6] + -0.0000*x[6]*x[7] 
          + -0.0000*x[6]*x[8] + 0.0000*x[7]*x[7] + 0.1267*x[7]*x[8] + -0.0000*x[8]*x[8])
    
    dx2 = (-0.0000 + 0.0000*x[0] + -0.0000*x[1] + -0.3791*x[2] + 0.0000*x[3] + -0.0000*x[4] 
          + 0.2335*x[5] + 0.0000*x[6] + -0.0000*x[7] + 0.0000*x[8] + 0.0000*x[0]*x[0] 
          + -0.0000*x[0]*x[1] + -0.0000*x[0]*x[2] + -0.0000*x[0]*x[3] + -0.0000*x[0]*x[4] 
          + -0.0000*x[0]*x[5] + 0.0000*x[0]*x[6] + -0.0000*x[0]*x[7] + -0.0000*x[0]*x[8] 
          + 0.0000*x[1]*x[1] + 0.0000*x[1]*x[2] + 0.0000*x[1]*x[3] + 0.0000*x[1]*x[4] 
          + -0.0000*x[1]*x[5] + -0.0000*x[1]*x[6] + 0.0000*x[1]*x[7] + -0.0000*x[1]*x[8] 
          + -0.0000*x[2]*x[2] + -0.0000*x[2]*x[3] + -0.0000*x[2]*x[4] + -0.0000*x[2]*x[5] 
          + -0.0000*x[2]*x[6] + 0.0000*x[2]*x[7] + 0.0000*x[2]*x[8] + -0.0000*x[3]*x[3] 
          + 0.0000*x[3]*x[4] + -0.0000*x[3]*x[5] + 0.0000*x[3]*x[6] + 0.0000*x[3]*x[7] 
          + 0.0000*x[3]*x[8] + -0.0000*x[4]*x[4] + 0.0000*x[4]*x[5] + 0.0000*x[4]*x[6] 
          + -0.0000*x[4]*x[7] + 0.0000*x[4]*x[8] + -0.0000*x[5]*x[5] + 0.0000*x[5]*x[6] 
          + 0.0000*x[5]*x[7] + 0.0000*x[5]*x[8] + 0.0000*x[6]*x[6] + -0.0000*x[6]*x[7] 
          + -0.3360*x[6]*x[8] + 0.0000*x[7]*x[7] + -0.0000*x[7]*x[8] + -0.0000*x[8]*x[8])
    
    dx3 = (-0.0000 + 0.3230*x[0] + -0.0000*x[1] + -16.9885*x[2] + 1.0333*x[3] + -2.5245*x[4] 
          + -21.6353*x[5] + -0.7242*x[6] + 0.0000*x[7] + 15.8862*x[8] + 0.0000*x[0]*x[0] 
          + 0.0000*x[0]*x[1] + -0.2060*x[0]*x[2] + -0.1509*x[0]*x[3] + 0.1689*x[0]*x[4] 
          + 0.4027*x[0]*x[5] + -0.0114*x[0]*x[6] + 0.1797*x[0]*x[7] + -0.0000*x[0]*x[8] 
          + 0.4760*x[1]*x[1] + 0.0000*x[1]*x[2] + 0.6134*x[1]*x[3] + 0.8522*x[1]*x[4] 
          + 0.4591*x[1]*x[5] + 1.1793*x[1]*x[6] + -0.4981*x[1]*x[7] + 0.0179*x[1]*x[8] 
          + 14.0228*x[2]*x[2] + -0.0000*x[2]*x[3] + 0.3106*x[2]*x[4] + 7.1862*x[2]*x[5] 
          + 0.4961*x[2]*x[6] + -0.8405*x[2]*x[7] + -17.0187*x[2]*x[8] + -0.1373*x[3]*x[3] 
          + -0.1125*x[3]*x[4] + -0.1727*x[3]*x[5] + 0.0053*x[3]*x[6] + 0.0633*x[3]*x[7] 
          + 0.0000*x[3]*x[8] + -0.0000*x[4]*x[4] + 0.2608*x[4]*x[5] + -0.1803*x[4]*x[6] 
          + -0.2294*x[4]*x[7] + 0.0246*x[4]*x[8] + -6.7851*x[5]*x[5] + 2.3323*x[5]*x[6] 
          + -3.7311*x[5]*x[7] + 9.3147*x[5]*x[8] + -0.1754*x[6]*x[6] + 0.5393*x[6]*x[7] 
          + 1.9092*x[6]*x[8] + -0.3464*x[7]*x[7] + -2.5161*x[7]*x[8] + -8.6107*x[8]*x[8])
    
    dx4 = (-0.0000 + 0.6521*x[0] + 0.0000*x[1] + 5.2048*x[2] + 1.7041*x[3] + -1.0492*x[4] 
          + -4.4808*x[5] + -0.7939*x[6] + 0.0202*x[7] + -23.7280*x[8] + 0.3976*x[0]*x[0] 
          + -0.4204*x[0]*x[1] + -0.0000*x[0]*x[2] + -0.9074*x[0]*x[3] + -1.3984*x[0]*x[4] 
          + 0.0909*x[0]*x[5] + -0.4789*x[0]*x[6] + 0.4115*x[0]*x[7] + -0.0000*x[0]*x[8] 
          + 0.6139*x[1]*x[1] + -0.0361*x[1]*x[2] + 0.6939*x[1]*x[3] + 0.8606*x[1]*x[4] 
          + 0.2734*x[1]*x[5] + 1.1959*x[1]*x[6] + -0.5214*x[1]*x[7] + 0.4275*x[1]*x[8] 
          + -10.8929*x[2]*x[2] + -0.2431*x[2]*x[3] + -0.0000*x[2]*x[4] + -4.0994*x[2]*x[5] 
          + 0.8093*x[2]*x[6] + -1.1496*x[2]*x[7] + 13.5315*x[2]*x[8] + -0.1843*x[3]*x[3] 
          + -0.5599*x[3]*x[4] + -1.1268*x[3]*x[5] + 0.1202*x[3]*x[6] + -0.0666*x[3]*x[7] 
          + -1.1398*x[3]*x[8] + -0.2352*x[4]*x[4] + -1.0998*x[4]*x[5] + 0.0000*x[4]*x[6] 
          + -0.0657*x[4]*x[7] + -1.0116*x[4]*x[8] + 4.4102*x[5]*x[5] + 3.6657*x[5]*x[6] 
          + -4.9323*x[5]*x[7] + -6.8917*x[5]*x[8] + -0.6002*x[6]*x[6] + 0.8231*x[6]*x[7] 
          + 2.7003*x[6]*x[8] + -0.5560*x[7]*x[7] + -3.7994*x[7]*x[8] + 7.0980*x[8]*x[8])
    
    dx5 = (0.0000 + -0.0000*x[0] + 0.0000*x[1] + 0.0000*x[2] + 0.0000*x[3] + 0.0000*x[4] 
          + 0.0000*x[5] + -0.0000*x[6] + 0.0000*x[7] + 0.0000*x[8] + 0.0000*x[0]*x[0] 
          + 0.0000*x[0]*x[1] + 0.0000*x[0]*x[2] + 0.0000*x[0]*x[3] + 0.0000*x[0]*x[4] 
          + -0.2195*x[0]*x[5] + -0.0000*x[0]*x[6] + 0.0000*x[0]*x[7] + 0.0744*x[0]*x[8] 
          + -0.0000*x[1]*x[1] + -0.0000*x[1]*x[2] + -0.0000*x[1]*x[3] + -0.0000*x[1]*x[4] 
          + 0.0000*x[1]*x[5] + 0.0000*x[1]*x[6] + -0.0000*x[1]*x[7] + -0.0000*x[1]*x[8] 
          + 0.0000*x[2]*x[2] + 0.0000*x[2]*x[3] + -0.0000*x[2]*x[4] + 0.0000*x[2]*x[5] 
          + 0.0000*x[2]*x[6] + -0.0000*x[2]*x[7] + -0.0000*x[2]*x[8] + -0.0000*x[3]*x[3] 
          + -0.0000*x[3]*x[4] + 0.0000*x[3]*x[5] + -0.0000*x[3]*x[6] + -0.0000*x[3]*x[7] 
          + -0.0000*x[3]*x[8] + 0.0000*x[4]*x[4] + 0.0000*x[4]*x[5] + 0.0000*x[4]*x[6] 
          + 0.0000*x[4]*x[7] + -0.0517*x[4]*x[8] + 0.0484*x[5]*x[5] + -0.0000*x[5]*x[6] 
          + 0.2143*x[5]*x[7] + -0.0000*x[5]*x[8] + -0.0000*x[6]*x[6] + 0.0000*x[6]*x[7] 
          + -0.0000*x[6]*x[8] + 0.0000*x[7]*x[7] + 0.0000*x[7]*x[8] + 0.0000*x[8]*x[8])
    
    dx6 = (0.0000 + 1.5716*x[0] + -0.3689*x[1] + -0.2580*x[2] + -0.3795*x[3] + -0.4308*x[4] 
          + 0.3048*x[5] + 0.6508*x[6] + -1.5700*x[7] + -0.0000*x[8] + -0.0000*x[0]*x[0] 
          + -0.7074*x[0]*x[1] + -0.2797*x[0]*x[2] + -0.5188*x[0]*x[3] + -0.6299*x[0]*x[4] 
          + -0.8793*x[0]*x[5] + 0.1576*x[0]*x[6] + -0.5122*x[0]*x[7] + -0.7761*x[0]*x[8] 
          + -1.0092*x[1]*x[1] + 0.6186*x[1]*x[2] + -0.6087*x[1]*x[3] + -0.8673*x[1]*x[4] 
          + 3.2850*x[1]*x[5] + -1.9200*x[1]*x[6] + -0.6004*x[1]*x[7] + 2.2485*x[1]*x[8] 
          + 0.2157*x[2]*x[2] + -0.0735*x[2]*x[3] + 0.0000*x[2]*x[4] + -0.0782*x[2]*x[5] 
          + -0.2367*x[2]*x[6] + 0.0000*x[2]*x[7] + -0.4513*x[2]*x[8] + 0.0000*x[3]*x[3] 
          + 0.1156*x[3]*x[4] + 0.0000*x[3]*x[5] + -0.4398*x[3]*x[6] + 0.0000*x[3]*x[7] 
          + -0.0000*x[3]*x[8] + 0.1992*x[4]*x[4] + -0.0000*x[4]*x[5] + -0.6961*x[4]*x[6] 
          + 0.0000*x[4]*x[7] + -0.0224*x[4]*x[8] + 0.3933*x[5]*x[5] + 0.1432*x[5]*x[6] 
          + 0.0824*x[5]*x[7] + 0.3871*x[5]*x[8] + -0.3320*x[6]*x[6] + -1.1779*x[6]*x[7] 
          + -0.2215*x[6]*x[8] + -0.0938*x[7]*x[7] + -0.0000*x[7]*x[8] + 0.0000*x[8]*x[8])
    
    dx7 = (0.0000 + -0.4781*x[0] + -0.8792*x[1] + 0.1180*x[2] + -0.0000*x[3] + -0.0754*x[4] 
          + -0.0909*x[5] + 1.0293*x[6] + 0.0504*x[7] + 0.0000*x[8] + 0.1894*x[0]*x[0] 
          + 1.4393*x[0]*x[1] + 0.2446*x[0]*x[2] + 0.7515*x[0]*x[3] + 1.0741*x[0]*x[4] 
          + 0.4978*x[0]*x[5] + 1.1477*x[0]*x[6] + -0.1790*x[0]*x[7] + 0.9670*x[0]*x[8] 
          + 1.3582*x[1]*x[1] + -0.0253*x[1]*x[2] + -0.0137*x[1]*x[3] + 0.0000*x[1]*x[4] 
          + 0.0914*x[1]*x[5] + 1.5322*x[1]*x[6] + 0.8508*x[1]*x[7] + 0.0508*x[1]*x[8] 
          + 0.3169*x[2]*x[2] + -0.0000*x[2]*x[3] + 0.3363*x[2]*x[4] + 0.0000*x[2]*x[5] 
          + -0.3982*x[2]*x[6] + 0.4660*x[2]*x[7] + -0.7885*x[2]*x[8] + -0.0000*x[3]*x[3] 
          + -0.0000*x[3]*x[4] + 0.2085*x[3]*x[5] + 0.5186*x[3]*x[6] + -0.4188*x[3]*x[7] 
          + 0.3557*x[3]*x[8] + 0.0000*x[4]*x[4] + 0.6940*x[4]*x[5] + 0.3331*x[4]*x[6] 
          + -0.2538*x[4]*x[7] + 0.7456*x[4]*x[8] + 0.6485*x[5]*x[5] + -0.0000*x[5]*x[6] 
          + -0.6960*x[5]*x[7] + -0.1073*x[5]*x[8] + 1.1794*x[6]*x[6] + -0.0578*x[6]*x[7] 
          + 0.1834*x[6]*x[8] + 0.6498*x[7]*x[7] + -0.3324*x[7]*x[8] + 0.0000*x[8]*x[8])
    
    dx8 = (-0.0000 + -0.0000*x[0] + -0.0000*x[1] + -0.0000*x[2] + 0.0000*x[3] + 0.0000*x[4] 
          + -0.0000*x[5] + -0.0000*x[6] + 0.0000*x[7] + 0.0000*x[8] + 0.0000*x[0]*x[0] 
          + -0.0000*x[0]*x[1] + -0.0000*x[0]*x[2] + -0.0000*x[0]*x[3] + 0.0000*x[0]*x[4] 
          + -0.0000*x[0]*x[5] + -0.0000*x[0]*x[6] + 0.0000*x[0]*x[7] + -0.0000*x[0]*x[8] 
          + 0.0000*x[1]*x[1] + 0.0000*x[1]*x[2] + 0.0000*x[1]*x[3] + -0.0000*x[1]*x[4] 
          + 0.4192*x[1]*x[5] + 0.0000*x[1]*x[6] + 0.0000*x[1]*x[7] + -0.0000*x[1]*x[8] 
          + 0.0000*x[2]*x[2] + -0.0000*x[2]*x[3] + 0.0000*x[2]*x[4] + 0.0000*x[2]*x[5] 
          + 0.0000*x[2]*x[6] + -0.0000*x[2]*x[7] + 0.0893*x[2]*x[8] + -0.0000*x[3]*x[3] 
          + 0.0000*x[3]*x[4] + -0.0000*x[3]*x[5] + -0.0000*x[3]*x[6] + -0.0000*x[3]*x[7] 
          + 0.0000*x[3]*x[8] + 0.0000*x[4]*x[4] + -0.0000*x[4]*x[5] + 0.0000*x[4]*x[6] 
          + 0.0000*x[4]*x[7] + 0.0000*x[4]*x[8] + 0.0000*x[5]*x[5] + -0.0000*x[5]*x[6] 
          + 0.0000*x[5]*x[7] + 0.0000*x[5]*x[8] + -0.0000*x[6]*x[6] + -0.0000*x[6]*x[7] 
          + 0.0000*x[6]*x[8] + -0.0000*x[7]*x[7] + 0.0000*x[7]*x[8] + 0.0067*x[8]*x[8])

    return [dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8]

# Initial conditions
x1, y1, z1 = 1.0, 0.0, 0.0
x2, y2, z2 = -0.5, 0.866, 0.0
x3, y3, z3 = -0.5, -0.866, 0.0

initial_state = [x1, y1, z1, x2, y2, z2, x3, y3, z3]

# Solve ODE
t_start = 0
t_end = 4.0
t_span = (t_start, t_end)
t_eval = np.linspace(t_start, t_end, 4000)

sol = solve_ivp(learned_three_body_odes, t_span, initial_state, t_eval=t_eval)

# Extract trajectories
body1 = sol.y[0:3]
body2 = sol.y[3:6]
body3 = sol.y[6:9]

os.makedirs('img/three_body', exist_ok=True)

# Create the 2D subplot layout once with both time series and x-y projection
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# First subplot (time series)
ax_time = axs[0]
ax_time.plot(sol.t, body1[0], 'r-', label='Body 1 (x)')
ax_time.plot(sol.t, body1[1], 'r--', label='Body 1 (y)')
ax_time.plot(sol.t, body1[2], 'r:', label='Body 1 (z)')
ax_time.plot(sol.t, body2[0], 'g-', label='Body 2 (x)')
ax_time.plot(sol.t, body2[1], 'g--', label='Body 2 (y)')
ax_time.plot(sol.t, body2[2], 'g:', label='Body 2 (z)')
ax_time.plot(sol.t, body3[0], 'b-', label='Body 3 (x)')
ax_time.plot(sol.t, body3[1], 'b--', label='Body 3 (y)')
ax_time.plot(sol.t, body3[2], 'b:', label='Body 3 (z)')
ax_time.set_xlabel('Time')
ax_time.set_ylabel('Position')
ax_time.set_title('Time Series of Body Positions')
ax_time.legend(loc='upper right', fontsize='small')
ax_time.grid(True)

# Second subplot (x-y projection)
ax_xy = axs[1]
ax_xy.plot(body1[0], body1[1], 'r-', label='Body 1')
ax_xy.plot(body2[0], body2[1], 'g-', label='Body 2')
ax_xy.plot(body3[0], body3[1], 'b-', label='Body 3')
ax_xy.scatter(body1[0, 0], body1[1, 0], c='red', s=100, marker='o')
ax_xy.scatter(body2[0, 0], body2[1, 0], c='green', s=100, marker='o')
ax_xy.scatter(body3[0, 0], body3[1, 0], c='blue', s=100, marker='o')
ax_xy.set_xlabel('X Position')
ax_xy.set_ylabel('Y Position')
ax_xy.set_title('X-Y Projection of Trajectories')
ax_xy.legend()
ax_xy.grid(True)

# Set fixed axis limits [-2, 2] for both axes while maintaining aspect ratio
ax_xy.set_xlim(-2, 2)
ax_xy.set_ylim(-2, 2)
ax_xy.set_aspect('equal')

plt.tight_layout()
plt.savefig('img/three_body/trajectory_plot.png')
plt.close()

# 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 3D trajectories
ax.plot(body1[0], body1[1], body1[2], 'r-', linewidth=1.5, label='Body 1')
ax.plot(body2[0], body2[1], body2[2], 'g-', linewidth=1.5, label='Body 2')
ax.plot(body3[0], body3[1], body3[2], 'b-', linewidth=1.5, label='Body 3')

ax.scatter(body1[0, 0], body1[1, 0], body1[2, 0], c='red', s=100, marker='o')
ax.scatter(body2[0, 0], body2[1, 0], body2[2, 0], c='green', s=100, marker='o')
ax.scatter(body3[0, 0], body3[1, 0], body3[2, 0], c='blue', s=100, marker='o')

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Three-Body System Trajectories')
ax.legend()
plt.savefig('img/three_body/3d_trajectory.png')
plt.close()

# Animation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(body1[0], body1[1], body1[2], 'r-', linewidth=0.5, alpha=0.3)
ax.plot(body2[0], body2[1], body2[2], 'g-', linewidth=0.5, alpha=0.3)
ax.plot(body3[0], body3[1], body3[2], 'b-', linewidth=0.5, alpha=0.3)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Three-Body System Animation')

body1_point, = ax.plot([], [], [], 'ro', markersize=8)
body2_point, = ax.plot([], [], [], 'go', markersize=8)
body3_point, = ax.plot([], [], [], 'bo', markersize=8)

# Trails
trail_length = 30
body1_trail, = ax.plot([], [], [], 'r-', linewidth=2)
body2_trail, = ax.plot([], [], [], 'g-', linewidth=2)
body3_trail, = ax.plot([], [], [], 'b-', linewidth=2)

# Text annotation for time
time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

def init():
    body1_point.set_data([], [])
    body1_point.set_3d_properties([])
    body2_point.set_data([], [])
    body2_point.set_3d_properties([])
    body3_point.set_data([], [])
    body3_point.set_3d_properties([])
    
    body1_trail.set_data([], [])
    body1_trail.set_3d_properties([])
    body2_trail.set_data([], [])
    body2_trail.set_3d_properties([])
    body3_trail.set_data([], [])
    body3_trail.set_3d_properties([])
    
    time_text.set_text('')
    
    return body1_point, body2_point, body3_point, body1_trail, body2_trail, body3_trail, time_text

def update(frame):
    frame = frame * 1
    if frame >= len(sol.t):
        frame = len(sol.t) - 1
        
    # Update points
    body1_point.set_data([body1[0, frame]], [body1[1, frame]])
    body1_point.set_3d_properties([body1[2, frame]])
    body2_point.set_data([body2[0, frame]], [body2[1, frame]])
    body2_point.set_3d_properties([body2[2, frame]])
    body3_point.set_data([body3[0, frame]], [body3[1, frame]])
    body3_point.set_3d_properties([body3[2, frame]])
    
    # Update trails
    start = max(0, frame - trail_length)
    body1_trail.set_data(body1[0, start:frame], body1[1, start:frame])
    body1_trail.set_3d_properties(body1[2, start:frame])
    body2_trail.set_data(body2[0, start:frame], body2[1, start:frame])
    body2_trail.set_3d_properties(body2[2, start:frame])
    body3_trail.set_data(body3[0, start:frame], body3[1, start:frame])
    body3_trail.set_3d_properties(body3[2, start:frame])
    
    time_text.set_text(f'Time: {sol.t[frame]:.2f}')
    
    ax.view_init(elev=30, azim=frame/20)
    
    return body1_point, body2_point, body3_point, body1_trail, body2_trail, body3_trail, time_text

ani = FuncAnimation(fig, update, frames=range(800), init_func=init, interval=20, blit=True)

ani.save("img/three_body/three_body_animation.gif", writer=PillowWriter(fps=30))
plt.close()

print("Visualization complete. Images and animations saved to 'img/three_body/' directory.")