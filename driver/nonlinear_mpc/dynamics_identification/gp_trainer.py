import time

import torch
import config
import matplotlib
import argparse
import numpy as np
import gpytorch
import matplotlib.pyplot as plt
from scipy import interpolate
from driver.nonlinear_mpc.dynamics_identification.torch_dynamics import DynamicBicycle, BatchIndependentGPDynamics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--visualize', type=bool, default=True)
    parser.add_argument('-r', '--resample', type=bool, default=False)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-l', '--lr', type=float, default=1e-1)
    parser.add_argument('-d', '--dataset', type=str, default="sine_input")
    args = parser.parse_args()

    indices = np.load("car_data/%s/filtered_start_indices.npy" % args.dataset)
    dataset = np.load("car_data/%s/filtered_odometry_data.npy" % args.dataset)
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
    inputs = torch.as_tensor(inputs, dtype=torch.float32).cuda()
    targets = torch.as_tensor(targets, dtype=torch.float32).cuda()

    bicycle_model = DynamicBicycle().cuda()
    bicycle_weights_path = "dynamic_bicycle_weights.pt"
    bicycle_model.load_state_dict(torch.load(bicycle_weights_path))

    with torch.no_grad():
        bicycle_out = bicycle_model(inputs)
        errors = targets - bicycle_out

    gp_weights_path = "gp_weights.pt"
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=6).cuda()
    indices = np.random.choice(np.arange(len(inputs)), int(len(inputs)/1))
    indices = torch.as_tensor(indices).cuda()
    print(len(indices))
    gp_model = BatchIndependentGPDynamics(inputs[indices], errors[indices], likelihood, 6).cuda()
    try:
        gp_model.load_state_dict(torch.load(gp_weights_path))
    except Exception as e:
        print("failed to load weights")
        torch.save(gp_model.state_dict(), gp_weights_path)

    if not args.visualize:
        likelihood.train()
        gp_model.train()
        optimizer = torch.optim.Adam(gp_model.parameters(), lr=args.lr)  # Includes GaussianLikelihood parameters
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
        for i in range(args.epochs):
            optimizer.zero_grad()
            output = gp_model(inputs[indices])
            loss = -mll(output, errors[indices])
            loss.backward()
            optimizer.step()
            torch.save(gp_model.state_dict(), gp_weights_path)
            print('Iter %d/%d - Loss: %.3f' % (i + 1, args.epochs, loss.item()))
    else:
        n_lin = 5
        linearization_points = np.random.choice(np.arange(len(inputs)), n_lin)
        linearization_points = torch.as_tensor(linearization_points).cuda()
        likelihood.eval()
        gp_model.eval()

        fun_outputs = []

        def batch_wrapper(input):
            outputs = likelihood(gp_model(input))
            fun_outputs.append(torch.stack([outputs.mean.detach(), outputs.stddev.detach()]))
            return torch.stack([outputs.mean, outputs.stddev]).sum(dim=1)

        means = torch.zeros([5, len(inputs), 6]).cuda() + bicycle_out
        stds = torch.zeros([5, len(inputs), 6]).cuda()
        selected_inputs = inputs[linearization_points]
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            _ = torch.autograd.functional.jacobian(batch_wrapper, selected_inputs, vectorize=True).moveaxis(-len(inputs.shape), 1)

        t1 = time.time_ns()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            jac = torch.autograd.functional.jacobian(batch_wrapper, selected_inputs, vectorize=True).moveaxis(-len(inputs.shape), 1)
            for i in range(n_lin):
                means[i] += fun_outputs[0][0, i] + (jac[0, i] @ (inputs - selected_inputs[i]).T).T
                stds[i] += fun_outputs[0][1, i] + (jac[1, i] @ (inputs - selected_inputs[i]).T).T
        t2 = time.time_ns()
        print("took %d ms" % ((t2-t1)/1e6))

        linearization_points = linearization_points.cpu().numpy()
        means = means.cpu().numpy()
        stds = stds.cpu().numpy()
        errors = errors.cpu().numpy()
        targets = targets.cpu().numpy()
        bicycle_out = bicycle_out.cpu().numpy()

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
            ax[idx].plot(scaler * targets[:, idx], label="target")
            ax[idx].plot(scaler * bicycle_out[:, idx], label="bicycle")
            for i in range(1):
                ax[idx].plot(scaler * means[i, :, idx], label="gp+bicycle %d" % i)
                ax[idx].plot(linearization_points[i], scaler*targets[linearization_points[i], idx], 'k*', markersize=10)
                ax[idx].fill_between(np.arange(len(means[i, :, idx])), means[i, :, idx] - 2*stds[i, :, idx], means[i, :, idx] + 2*stds[i, :, idx], alpha=0.5)
            ax[idx].legend()

        plt.show()