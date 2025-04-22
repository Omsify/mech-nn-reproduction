import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.autograd import gradcheck


from torch.utils.data import Dataset, DataLoader
from scipy.integrate import odeint

from extras.source import write_source_files, create_log_dir

from solver.ode_layer import ODEINDLayer
import discovery.basis as B
import ipdb
import extras.logger as logger
import os

from scipy.special import logit
import torch.nn.functional as F
from tqdm import tqdm
import discovery.plot as P


log_dir, run_id = create_log_dir(root='logs')
write_source_files(log_dir)
L = logger.setup(log_dir, stdout=False)

DBL = True
dtype = torch.float64 if DBL else torch.float32
STEP = 0.01
cuda = True
T = 10000
n_step_per_batch = 50
batch_size = 512
# Weights less than threshold (absolute) are set to 0 after each optimization step
threshold = 0.1


class ThreeBodyDataset(Dataset):
    def __init__(self, n_step_per_batch=100, n_step=1000):
        self.n_step_per_batch = n_step_per_batch
        self.n_step = n_step
        self.end = n_step * STEP
        x_train = self.generate()

        self.down_sample = 1

        self.x_train = torch.tensor(x_train, dtype=dtype)

        # Create basis for some stats. Actual basis is in the model
        # Three coordinates for each of the three bodies
        basis, basis_vars = B.create_library(x_train, polynomial_order=2, use_trig=False, constant=True)
        self.basis = torch.tensor(basis)
        self.basis_vars = basis_vars
        self.n_basis = self.basis.shape[1]

    def generate(self):
        G = 1.0
        # Masses
        m1, m2, m3 = 1.0, 1.0, 1.0
        masses = np.array([m1, m2, m3])
        
        def f(state, t):
            # State = (positions and velocities of 3 bodies)
            r = state.reshape(3, 6)  # Reshape to 3 bodies * (x,y,z,vx,vy,vz)
            
            pos = r[:, :3]
            vel = r[:, 3:] 
            
            acc = np.zeros((3, 3))
            
            # Calculate accelerations
            for i in range(3):
                for j in range(3):
                    if i != j:
                        r_ij = pos[j] - pos[i]
                        r_ij_mag = np.sqrt(np.sum(r_ij**2))
                        acc[i] += G * masses[j] * r_ij / r_ij_mag**3
            
            derivatives = np.zeros_like(r)
            derivatives[:, :3] = vel
            derivatives[:, 3:] = acc
            
            return derivatives.flatten()

        # Initial conditions (triangle)
        x1, y1, z1 = 1.0, 0.0, 0.0
        vx1, vy1, vz1 = 0.0, 0.5, 0.0

        x2, y2, z2 = -0.5, 0.866, 0.0 
        vx2, vy2, vz2 = -0.433, -0.25, 0.0

        x3, y3, z3 = -0.5, -0.866, 0.0
        vx3, vy3, vz3 = 0.433, -0.25, 0.0
        
        state0 = np.array([
            x1, y1, z1, vx1, vy1, vz1,
            x2, y2, z2, vx2, vy2, vz2,
            x3, y3, z3, vx3, vy3, vz3
        ])
        
        time_steps = np.linspace(0, self.end, self.n_step)
        self.time_steps = time_steps
        
        trajectory = odeint(f, state0, time_steps)
        
        x_train = trajectory[:, :9]
        
        return x_train

    def __len__(self):
        return (self.n_step-self.n_step_per_batch)//self.down_sample

    def __getitem__(self, idx):
        i = idx*self.down_sample
        d = self.x_train[i:i+self.n_step_per_batch]
        return i, d


ds = ThreeBodyDataset(n_step=T, n_step_per_batch=n_step_per_batch)
train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

def plot_three_body(data, save_path=None):
    """
    Plot 3D trajectory of the three-body system
    data: tensor or numpy array of shape [time_steps, 9] 
          where 9 is (x,y,z) for each of the 3 bodies
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Coordinates
    body1 = data[:, 0:3]
    body2 = data[:, 3:6]
    body3 = data[:, 6:9]
    
    # Trajectories
    ax.plot(body1[:, 0], body1[:, 1], body1[:, 2], 'r-', linewidth=1, label='Body 1')
    ax.plot(body2[:, 0], body2[:, 1], body2[:, 2], 'g-', linewidth=1, label='Body 2')
    ax.plot(body3[:, 0], body3[:, 1], body3[:, 2], 'b-', linewidth=1, label='Body 3')
    
    # Starting points
    ax.scatter(body1[0, 0], body1[0, 1], body1[0, 2], c='r', marker='o', s=100)
    ax.scatter(body2[0, 0], body2[0, 1], body2[0, 2], c='g', marker='o', s=100)
    ax.scatter(body3[0, 0], body3[0, 1], body3[0, 2], c='b', marker='o', s=100)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Three-Body Problem Trajectory')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

plot_three_body(ds.x_train, os.path.join(log_dir, 'train.pdf'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, bs, n_step, n_step_per_batch, n_basis, device=None, **kwargs):
        super().__init__()

        self.n_step = n_step
        self.order = 2
        # State dimension 
        self.bs = bs
        self.device = device
        self.n_iv = 1
        self.n_ind_dim = 9
        self.n_step_per_batch = n_step_per_batch

        self.n_basis = ds.n_basis

        self.init_xi = torch.randn((1, self.n_basis, self.n_ind_dim), dtype=dtype).to(device)

        self.mask = torch.ones_like(self.init_xi).to(device)

        # Step size is fixed. Make this a parameter for learned step
        self.step_size = (logit(0.01)*torch.ones(1,1,1))
        self.xi = nn.Parameter(self.init_xi.clone())
        self.param_in = nn.Parameter(torch.randn(1,64))

        init_coeffs = torch.rand(1, self.n_ind_dim, 1, 2, dtype=dtype)
        self.init_coeffs = nn.Parameter(init_coeffs)
        
        self.ode = ODEINDLayer(bs=bs, order=self.order, n_ind_dim=self.n_ind_dim, n_step=self.n_step_per_batch, solver_dbl=True, double_ret=True,
                                n_iv=self.n_iv, n_iv_steps=1, gamma=0.05, alpha=0, **kwargs)

        self.param_net = nn.Sequential(
            nn.Linear(64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.n_basis*self.n_ind_dim)
        )

        self.net = nn.Sequential(
            nn.Linear(self.n_step_per_batch*self.n_ind_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.n_step_per_batch*self.n_ind_dim)
        )
    
    def reset_params(self):
        # Reset basis weights to random values
        self.xi.data = torch.randn_like(self.init_xi)

    def update_mask(self, mask):
        self.mask = self.mask*mask
    
    def get_xi(self):
        xi = self.param_net(self.param_in)
        xi = xi.reshape(self.init_xi.shape)
        return xi

    def forward(self, index, net_iv):
        # Apply mask
        xi = self.get_xi()
        xi = self.mask*xi
        _xi = xi
        xi = xi.repeat(self.bs, 1, 1)

        var = self.net(net_iv.reshape(self.bs, -1))
        var = var.reshape(self.bs, self.n_step_per_batch, self.n_ind_dim)

        # Create basis
        var_basis, _ = B.create_library_tensor_batched(var, polynomial_order=2, use_trig=False, constant=True)

        rhs = var_basis@xi
        rhs = rhs.permute(0, 2, 1)

        z = torch.zeros(1, self.n_ind_dim, 1, 1).type_as(net_iv)
        o = torch.ones(1, self.n_ind_dim, 1, 1).type_as(net_iv)

        coeffs = torch.cat([z, o, z], dim=-1)
        coeffs = coeffs.repeat(self.bs, 1, self.n_step_per_batch, 1)

        init_iv = var[:, 0]

        steps = self.step_size.repeat(self.bs, self.n_ind_dim, self.n_step_per_batch-1).type_as(net_iv)
        steps = torch.sigmoid(steps)

        x0, x1, x2, eps, steps = self.ode(coeffs, rhs, init_iv, steps)
        x0 = x0.permute(0, 2, 1)

        return x0, steps, eps, var, _xi

model = Model(bs=batch_size, n_step=T, n_step_per_batch=n_step_per_batch, n_basis=ds.n_basis, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

if DBL:
    model = model.double()
model = model.to(device)


def print_eq(stdout=False):
    # Print learned equation
    xi = model.get_xi()
    repr_dict = B.basis_repr(xi*model.mask, ds.basis_vars)
    code = []
    for k, v in repr_dict.items():
        L.info(f'{k} = {v}')
        if stdout:
            print(f'{k} = {v}')
        code.append(f'{v}')
    return code

def simulate(gen_code):
    # Simulate learned equation
    def f(state, t):
        x0, x1, x2, x3, x4, x5, x6, x7, x8 = state

        dx0 = eval(gen_code[0])  # 1, x
        dx1 = eval(gen_code[1])  # 1, y
        dx2 = eval(gen_code[2])  # 1, z
        dx3 = eval(gen_code[3])  # ...
        dx4 = eval(gen_code[4])
        dx5 = eval(gen_code[5])
        dx6 = eval(gen_code[6])
        dx7 = eval(gen_code[7])
        dx8 = eval(gen_code[8])

        return dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8
        
    state0 = ds.x_train[0].numpy()
    time_steps = np.linspace(0, T*STEP, T)

    x_sim = odeint(f, state0, time_steps)
    return x_sim

def train():
    """Optimize and threshold cycle"""
    model.reset_params()

    max_iter = 10
    for step in range(max_iter):
        print(f'Optimizer iteration {step}/{max_iter}')

        # Threshold
        if step > 0:
            xi = model.get_xi()
            mask = (xi.abs() > threshold).float()

            L.info(xi)
            L.info(xi*model.mask)
            L.info(model.mask)
            L.info(model.mask*mask)

        code = print_eq(stdout=True)
        # Simulate and plot

        x_sim = simulate(code)
        plot_three_body(x_sim, os.path.join(log_dir, f'sim_{step}.pdf'))

        # Set mask
        if step > 0:
            model.update_mask(mask)
            model.reset_params()

        optimize()


def optimize(nepoch=400):
    with tqdm(total=nepoch) as pbar:
        for epoch in range(nepoch):
            pbar.update(1)
            for i, (index, batch_in) in enumerate(train_loader):
                batch_in = batch_in.to(device)

                optimizer.zero_grad()
                x0, steps, eps, var, xi = model(index, batch_in)

                x_loss = (x0 - batch_in).pow(2).mean()
                loss = x_loss + (var - batch_in).pow(2).mean()
                
                loss.backward()
                optimizer.step()

            xi = xi.detach().cpu().numpy()
            meps = eps.max().item()
            L.info(f'run {run_id} epoch {epoch}, loss {loss.item()} max eps {meps} xloss {x_loss} ')
            print(f'basis\n {xi}')
            pbar.set_description(f'run {run_id} epoch {epoch}, loss {loss.item()} max eps {meps} xloss {x_loss} ')


if __name__ == "__main__":
    train()
    print_eq()