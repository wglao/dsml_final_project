import os
import torch
import numpy as np
import wandb
import pickle as pkl


# pointwise MSE
# use dt argument for api matching
def timeseries_MSE_loss(pred_y, y, dt=None):
    loss_value = torch.mean((pred_y - y) ** 2)
    return loss_value


# L2 loss using Lagrange 1st order interpolant
# default to daily prediction
def timeseries_L2_loss(pred_y, y, dt: float = 1.0):
    time_integrate = dt * torch.ones_like(y)
    time_integrate[0] = 0.5 * dt
    time_integrate[-1] = 0.5 * dt
    loss_integral = time_integrate @ ((pred_y - y) ** 2)
    loss_value = torch.sqrt(loss_integral)
    return loss_value


# H1 loss using C1 and P0 interpolant
# default to daily prediction
def timeseries_H1_loss(pred_y, y, dt: float = 1.0):
    time_integrate_1 = dt * torch.ones_like(y)
    time_integrate_1[0] = 0.5 * dt
    time_integrate_1[-1] = 0.5 * dt
    time_integrate_2 = dt * torch.ones_like(y)[1:]
    
    L2_value = torch.sqrt(time_integrate_1 @ ((pred_y - y) ** 2))
    H1_value = L2_value + torch.sqrt(
        time_integrate_2 @ ((torch.diff(pred_y - y)) ** 2))
    return H1_value


def naive_mlp_epoch(
    model,
    optimizer,
    train_loader,
    loss_fn: callable = timeseries_MSE_loss,
    dt: float = 1.0,
):
    model.train(True)
    running_loss = 0.0
    last_loss = 0.0

    for batch in iter(train_loader):
        xs, ys = batch
        optimizer.zero_grad()
        for x, y in zip(xs,ys):
            pred_y = model(x)

            loss = loss_fn(pred_y, y, dt)
            loss.backward()
        optimizer.step()

        last_loss = loss.item()
        running_loss += loss.item()

    return last_loss, running_loss / len(train_loader)

def noisy_mlp_epoch(
    model,
    optimizer,
    train_loader,
    loss_fn: callable = timeseries_MSE_loss,
    dt: float = 1.0,
    noise_variance: float = 0.01
):
    model.train(True)
    running_loss = 0.0
    last_loss = 0.0

    for batch in iter(train_loader):
        xs, ys = batch
        optimizer.zero_grad()
        for x, y in zip(xs,ys):
            noise = noise_variance * torch.randn(x.shape[0])
            noisy_x = x + noise
            pred_y = model(noisy_x)

            loss = loss_fn(pred_y, y, dt)
            loss.backward()
        optimizer.step()

        last_loss = loss.item()
        running_loss += loss.item()

    return last_loss, running_loss / len(train_loader)



def naive_mlp_test(
    model, test_loader, loss_fn: callable = timeseries_MSE_loss, dt: float = 1.0
):
    running_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in iter(test_loader):
            xs, ys = batch
            for x, y in zip(xs,ys):
                pred_y = model(x)
                loss = loss_fn(pred_y, y, dt)
                running_loss += loss.item()
            test_loss = running_loss / len(test_loader)
        return test_loss

def naive_train(
    model,
    optimizer,
    train_loader,
    test_loader,
    num_epochs,
    loss_fn: callable = timeseries_MSE_loss,
    log_wandb: bool = False,
    name: str = "test",
    save_dir: str = "saved_models",
    print_every: int = 100,
    dt: float = 1.0
):
    if log_wandb:
        wandb.init(project="DSML Final", name=name)

    min_error = 1e5
    for epoch in range(num_epochs):
        last_loss, running_loss = naive_mlp_epoch(
            model, optimizer, train_loader, loss_fn, dt
        )
        test_loss = naive_mlp_test(model, test_loader, loss_fn)

        if (epoch == num_epochs - 1) or ((epoch % print_every) == 0):
            if log_wandb:
                wandb.log(
                    {
                        "Epoch": epoch,
                        "Train Loss": running_loss,
                        "Test Loss": test_loss,
                        "Last Loss": last_loss,
                    }
                )
            print(
                "Epoch: {}   Train: {} ({})    Test: {}".format(
                    epoch, running_loss, last_loss, test_loss
                )
            )

        if test_loss < min_error:
            min_error = test_loss
            filename = os.path.join(save_dir, name + ".pkl")
            if os.path.isfile(filename):
                os.rename(filename, filename + "-old")
            with open(filename, "wb") as f:
                pkl.dump(model, f)

            opt_filename = os.path.join(save_dir, name + "_opt.pkl")
            if os.path.isfile(opt_filename):
                os.rename(opt_filename, opt_filename + "-old")
            with open(opt_filename, "wb") as f:
                pkl.dump(optimizer, f)


def noisy_train(
    model,
    optimizer,
    train_loader,
    test_loader,
    num_epochs,
    loss_fn: callable = timeseries_MSE_loss,
    log_wandb: bool = False,
    name: str = "test",
    print_every: int = 100,
    save_dir: str = "saved_models",
    dt: float = 1.0,
    noise_variance: float = 0.01,
):
    if log_wandb:
        wandb.init(project="DSML Final", name=name)

    min_error = 1e5
    for epoch in range(num_epochs):
        last_loss, running_loss = noisy_mlp_epoch(
            model, optimizer, train_loader, loss_fn, dt, noise_variance
        )
        test_loss = naive_mlp_test(model, test_loader, loss_fn)

        if (epoch == num_epochs - 1) or ((epoch % print_every) == 0):
            if log_wandb:
                wandb.log(
                    {
                        "Epoch": epoch,
                        "Train Loss": running_loss,
                        "Test Loss": test_loss,
                        "Last Loss": last_loss,
                    }
                )
            print(
                "Epoch: {}   Train: {} ({})    Test: {}".format(
                    epoch, running_loss, last_loss, test_loss
                )
            )

        if test_loss < min_error:
            min_error = test_loss
            filename = os.path.join(save_dir, name + ".pkl")
            if os.path.isfile(filename):
                os.rename(filename, filename + "-old")
            with open(filename, "wb") as f:
                pkl.dump(model, f)

            opt_filename = os.path.join(save_dir, name + "_opt.pkl")
            if os.path.isfile(opt_filename):
                os.rename(opt_filename, opt_filename + "-old")
            with open(opt_filename, "wb") as f:
                pkl.dump(optimizer, f)
