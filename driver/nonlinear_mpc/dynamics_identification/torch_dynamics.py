from torch.utils.data import Dataset
import config
import torch


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


class DynamicBicycle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inertia = torch.nn.Parameter(torch.randn(1))
        self.Bf = torch.nn.Parameter(torch.randn(1))
        self.Cf = torch.nn.Parameter(torch.randn(1))
        self.Df = torch.nn.Parameter(torch.randn(1))
        self.Br = torch.nn.Parameter(torch.randn(1))
        self.Cr = torch.nn.Parameter(torch.randn(1))
        self.Dr = torch.nn.Parameter(torch.randn(1))
        self.Tm = torch.nn.Parameter(torch.randn(1))
        self.Tr0 = torch.nn.Parameter(torch.randn(1))
        self.Tr2 = torch.nn.Parameter(torch.randn(1))

    def forward(self, tensor):
        hdg = tensor[:, 0]
        vx = tensor[:, 1]
        vy = tensor[:, 2]
        w = tensor[:, 3]
        steer = tensor[:, 4]
        throttle = tensor[:, 5]
        steer_dot = tensor[:, 6]

        ar = torch.arctan2(vy - config.car_lr*w, vx)
        Fry = self.Dr * torch.sin(self.Cr * torch.arctan(self.Br * ar))

        af = torch.arctan2(vy + config.car_lf*w, vx) - steer*config.car_max_steer
        Ffy = self.Df * torch.sin(self.Cf * torch.arctan(self.Bf * af))

        Frx = self.Tm*throttle - self.Tr0 - self.Tr2 * vx ** 2

        out_d = torch.vstack([
             vx * torch.cos(hdg) + vy * torch.sin(hdg),
             vx * torch.sin(hdg) + vy * torch.cos(hdg),
             w,
             1 / config.car_mass * Frx,
             1 / config.car_mass * (Ffy * torch.cos(steer * config.car_max_steer) + Fry),
             1 / self.inertia * (Ffy * config.car_lf * torch.cos(steer * config.car_max_steer) - Fry * config.car_lr)
        ]).T

        out_k = torch.vstack([
            vx * torch.cos(hdg) + vy * torch.sin(hdg),
            vx * torch.sin(hdg) + vy * torch.cos(hdg),
            w,
            Frx / config.car_mass,
            (steer_dot*config.car_max_steer*vx + steer*config.car_max_steer*(Frx/config.car_mass)) * (config.car_lr/(config.car_lr + config.car_lf)),
            (steer_dot*config.car_max_steer*vx + steer*config.car_max_steer*(Frx/config.car_mass)) * (1 / (config.car_lr + config.car_lf)),
        ]).T

        vb_min = config.blend_min_speed
        vb_max = config.blend_max_speed
        lam = torch.minimum(torch.maximum((vx-vb_min) / (vb_max-vb_min), torch.zeros_like(vx)), torch.ones_like(vx))

        return (lam*out_d.T).T + ((1-lam)*out_k.T).T
