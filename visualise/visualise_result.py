import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

def learned_odes_extended(t, x):
    x0, x1, x2 = x
    dx0 = (
        -0.0000
        - 8.4991 * x0
        + 8.9277 * x1
        - 0.0000 * x2
        - 0.0025 * x0**2
        + 0.0021 * x0 * x1
        - 0.0358 * x0 * x2
        - 0.0000 * x1**2
        + 0.0227 * x1 * x2
        + 0.0000 * x2**2
    )
    dx1 = (
        1.0575
        + 20.0465 * x0
        + 4.2891 * x1
        - 0.0547 * x2
        + 0.0154 * x0**2
        - 0.0108 * x0 * x1
        - 0.7631 * x0 * x2
        + 0.0000 * x1**2
        - 0.1551 * x1 * x2
        - 0.0000 * x2**2
    )
    dx2 = (
        0.6265
        + 0.0052 * x0
        - 0.0149 * x1
        - 2.6752 * x2
        + 0.0070 * x0**2
        + 0.9449 * x0 * x1
        + 0.0000 * x0 * x2
        + 0.0414 * x1**2
        - 0.0000 * x1 * x2
        - 0.0000 * x2**2
    )
    return [dx0, dx1, dx2]

# Solve ODE
t_start = 0
t_end = 40
t_span = (t_start, t_end)
t_eval = np.linspace(t_start, t_end, 2000)
initial_state = [1.0, 0.0, -1.0]
sol_ext = solve_ivp(learned_odes_extended, t_span, initial_state, t_eval=t_eval)
x0, x1, x2 = sol_ext.y

initial_state = [1.0, 0.0, -1.0]

sol_ext = solve_ivp(learned_odes_extended, t_span, initial_state, t_eval=t_eval)

plt.figure(figsize=(12, 5))
plt.plot(sol_ext.t, sol_ext.y[0], label='x0')
plt.plot(sol_ext.t, sol_ext.y[1], label='x1')
plt.plot(sol_ext.t, sol_ext.y[2], label='x2')
plt.xlabel('Time')
plt.ylabel('State values')
plt.legend()
plt.title(f'Time Series of x0, x1, x2 over [{t_start}, {t_end}]')
plt.grid(True)
plt.savefig('img/reproduction/trajectory_timeseries.png')
plt.close()

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol_ext.y[0], sol_ext.y[1], sol_ext.y[2], lw=1)
ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('x2')
ax.set_title(f'Phase-Space Trajectory over [{t_start}, {t_end}]')
line, = ax.plot([], [], [], lw=1.5)

def update(frame):
    frame = int(frame)
    ax.view_init(elev=30, azim=frame)
    line.set_data(x0[:frame], x1[:frame])
    line.set_3d_properties(x2[:frame])
    return line,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 360, 240), interval=50, blit=True)

ani.save("img/reproduction/trajectory_animation.gif", writer=PillowWriter(fps=20))
plt.close()