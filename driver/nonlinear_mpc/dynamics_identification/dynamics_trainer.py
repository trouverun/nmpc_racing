import torch
import config
import matplotlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from torch.utils.data import SubsetRandomSampler, DataLoader
from driver.nonlinear_mpc.dynamics_identification.torch_dynamics import DynamicBicycle, DynamicsDataset, KinematicBicycle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--visualize', type=bool, default=True)
    parser.add_argument('-r', '--resample', type=bool, default=False)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-l', '--lr', type=float, default=1e+1)
    parser.add_argument('-d', '--dataset', type=str, default="12_01_2022__17_30_23_dynamic_bicycle")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_model = DynamicBicycle()

    weights_path = "weights/dynamic_bicycle_weights.pt"
    try:
        w = torch.load(weights_path)
        for (k, v) in w.items():
            print("%s = %.8f" % (k, v))
        torch_model.load_state_dict(w)
    except:
        torch.save(torch_model.state_dict(), weights_path)

    indices = np.load("%s/%s/filtered_start_indices.npy" % (config.dynamics_data_folder, args.dataset))
    dataset = np.load("%s/%s/filtered_odometry_data.npy" % (config.dynamics_data_folder, args.dataset))

    inputs = []
    targets = []

    for i in range(len(indices)):
        start = indices[i]
        if i == len(indices)-1:
            end = len(dataset)
        else:
            end = indices[i+1]

        if args.resample:
            dt = 0.01
            new_timestamps = np.arange(0, dataset[end-1, 0], dt)
            resampled_dataset = np.c_[new_timestamps]
            for j in range(1, dataset.shape[1]):
                f = interpolate.interp1d(dataset[start:end, 0], dataset[start:end, j])
                resampled = f(new_timestamps)
                resampled_dataset = np.c_[resampled_dataset, resampled]
            track_data = resampled_dataset

        track_data = dataset[start:end]
        tmp_inputs = track_data[:-1, [3, 4, 5, 6, 11, 13, 22]]
        tmp_targets = track_data[1:, [18, 19, 6, 7, 8, 9]]

        inputs.extend(tmp_inputs)
        targets.extend(tmp_targets)

    inputs = np.asarray(inputs)
    targets = np.asarray(targets)

    if not args.visualize:
        torch_model = torch_model.to(device)

        # Create dataset and data loaders
        dataset = DynamicsDataset(torch.as_tensor(inputs), torch.as_tensor(targets))
        all_indices = np.arange(len(inputs))
        train_indices = np.random.choice(all_indices, int(0.95 * len(inputs)), False)
        train_sampler = SubsetRandomSampler(train_indices)
        valid_indices = np.setdiff1d(all_indices, train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=1024)
        valid_loader = DataLoader(dataset, sampler=valid_sampler, batch_size=1024)

        epochs = args.epochs
        lr = args.lr

        optimizer = torch.optim.AdamW(torch_model.parameters(), lr=lr, weight_decay=0, betas=(0.90, 0.999))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, max_lr=lr, total_steps=None,
            epochs=epochs, steps_per_epoch=len(train_sampler), pct_start=0.3,
            anneal_strategy='cos', cycle_momentum=True,
            base_momentum=0.85, max_momentum=0.95, div_factor=25.0,
            final_div_factor=1000.0, last_epoch=-1
        )

        loss_fn = torch.nn.MSELoss(reduction="sum")

        w = torch.tensor([1, 1, 22.5/np.pi, 1, 1, 22.5/np.pi]).to(device)
        w = torch.ones(6).to(device)
        for epoch in range(epochs):
            torch_model.train()
            train_loss = 0
            for data in train_loader:
                inputs, labels = data[0].to(device, dtype=torch.float), data[1].to(device, dtype=torch.float)
                optimizer.zero_grad()
                outputs = torch_model(inputs)
                loss = loss_fn(w*outputs, w*labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                # scheduler.step()

            torch_model.eval()
            val_loss = 0
            with torch.no_grad():
                for data in valid_loader:
                    inputs, labels = data[0].to(device, dtype=torch.float), data[1].to(device, dtype=torch.float)
                    outputs = torch_model(inputs)
                    loss = loss_fn(w*outputs, w*labels)
                    val_loss += loss.item()

            torch.save(torch_model.state_dict(), weights_path)
            print("Epoch %3d: Train loss %.8f, Validation loss %.8f" % (epoch, train_loss / len(train_sampler), val_loss / len(valid_sampler)))
    else:
        torch_model = torch_model.to(device)
        inputs = torch.as_tensor(inputs, dtype=torch.float32).to("cuda:0")
        targets = torch.as_tensor(targets, dtype=torch.float32).to("cuda:0")
        with torch.no_grad():
            outputs = torch_model(inputs)
        targets = targets.cpu().numpy()
        outputs = outputs.cpu().numpy()

        titles = [
            ("vx", 0),
            ("vy", 1),
            ("w", 2),
            ("ax", 3),
            ("ay", 4),
            ("dw", 5),
        ]

        matplotlib.use('QtAgg')
        fig, ax = plt.subplots(len(titles), 1, figsize=(14, 10))
        fig.tight_layout()
        inputs = inputs.cpu().numpy()

        for title, idx in titles:
            ax[idx].set_title(title)
            scaler = 1
            if title in ['w', 'dw']:
                scaler = 180 / np.pi
            ax[idx].plot(scaler * targets[:, idx], label="data")
            ax[idx].plot(scaler * outputs[:, idx], label="output")
            if title == 'ax':
                ax[idx].plot(10*inputs[:, 5], label="throttle")
            if title == 'dw':
                ax[idx].plot(inputs[:, 1], label="vx")
                ax[idx].plot(scaler*config.car_max_steer*inputs[:, 4], label="steer")
            ax[idx].legend()

        plt.show()