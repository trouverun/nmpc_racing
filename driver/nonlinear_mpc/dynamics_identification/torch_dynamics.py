from torch.utils.data import Dataset
import config
import torch
import gpytorch


class DynamicsDataset(Dataset):
    def __init__(self, x, y):
        assert len(x) == len(y)
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x[idx], self.y[idx]


class KinematicBicycle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_steer = torch.nn.Parameter(torch.randn(1))
        self.max_acceleration = torch.nn.Parameter(torch.randn(1))
        self.rolling_resistance = torch.nn.Parameter(torch.randn(1))
        self.drag = torch.nn.Parameter(torch.randn(1))

    def forward(self, tensor):
        hdg = tensor[:, 0]
        v = tensor[:, 1]
        slip = tensor[:, 2]
        steer = tensor[:, 3]
        throttle = tensor[:, 4]

        out_k = torch.vstack([
            v * torch.cos(hdg + slip),  # x
            v * torch.sin(hdg + slip),  # y
            v / config.car_lr * torch.sin(slip),  # psi
            throttle * self.max_acceleration - self.rolling_resistance - self.drag * v ** 2,
            torch.arctan2(config.car_lr * torch.tan(steer * self.max_steer), torch.tensor([config.car_lf + config.car_lr], requires_grad=True).cuda()),  # slip
        ]).T

        return out_k

class DynamicBicycle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.car_inertia = torch.nn.Parameter(torch.randn(1))
        self.wheel_Bf = torch.nn.Parameter(torch.randn(1))
        self.wheel_Cf = torch.nn.Parameter(torch.randn(1))
        self.wheel_Df = torch.nn.Parameter(torch.randn(1))
        self.wheel_Br = torch.nn.Parameter(torch.randn(1))
        self.wheel_Cr = torch.nn.Parameter(torch.randn(1))
        self.wheel_Dr = torch.nn.Parameter(torch.randn(1))
        self.car_Tm = torch.nn.Parameter(torch.randn(1))
        self.car_Tr0 = torch.nn.Parameter(torch.randn(1))
        self.car_Tr2 = torch.nn.Parameter(torch.randn(1))

    def forward(self, tensor):
        hdg = tensor[:, 0]
        vx = tensor[:, 1]
        vy = tensor[:, 2]
        w = tensor[:, 3]
        steer = tensor[:, 4]
        throttle = tensor[:, 5]
        steer_dot = tensor[:, 6]

        ar = torch.arctan2(vy - config.car_lr*w, vx)
        Fry = self.wheel_Dr * torch.sin(self.wheel_Cr * torch.arctan(self.wheel_Br * ar))

        af = torch.arctan2(vy + config.car_lf*w, vx) - steer*config.car_max_steer
        Ffy = self.wheel_Df * torch.sin(self.wheel_Cf * torch.arctan(self.wheel_Bf * af))

        Frx = self.car_Tm*throttle - self.car_Tr0 - self.car_Tr2 * vx ** 2

        out_d = torch.vstack([
             vx * torch.cos(hdg) + vy * torch.sin(hdg),
             vx * torch.sin(hdg) + vy * torch.cos(hdg),
             w,
             1 / config.car_mass * Frx,
             1 / config.car_mass * (Ffy * torch.cos(steer * config.car_max_steer) + Fry),
             1 / self.car_inertia * (Ffy * config.car_lf * torch.cos(steer * config.car_max_steer) - Fry * config.car_lr)
        ]).T

        out_k = torch.vstack([
            vx * torch.cos(hdg) + vy * torch.sin(hdg),
            vx * torch.sin(hdg) + vy * torch.cos(hdg),
            w,
            Frx / config.car_mass,
            (steer_dot*config.car_max_steer*vx + steer*config.car_max_steer*(Frx/config.car_mass)) * (config.car_lr / (config.car_lr + config.car_lf)),
            (steer_dot*config.car_max_steer*vx + steer*config.car_max_steer*(Frx/config.car_mass)) * (1 / (config.car_lr + config.car_lf)),
        ]).T

        vb_min = config.blend_min_speed
        vb_max = config.blend_max_speed
        lam = torch.minimum(torch.maximum((vx-vb_min) / (vb_max-vb_min), torch.zeros_like(vx)), torch.ones_like(vx))

        return (lam*out_d.T).T + ((1-lam)*out_k.T).T


class BatchIndependentGPDynamics(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_out):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([n_out]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([n_out])),
            batch_shape=torch.Size([n_out])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )