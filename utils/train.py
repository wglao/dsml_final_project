import os
import torch
import torch.nn.functional as F
import jax
import jax.numpy as jnp
import jax.random as jrd
import equinox as eqx
import numpy as np
import wandb
import pickle as pkl
import optax as opx


# pointwise MSE
# use dt argument for api matching
def timeseries_MSE_loss(pred_y, y, dt=None):
    loss_value = torch.sum((pred_y - y) ** 2)
    return loss_value


# L2 loss using Lagrange 1st order interpolant
# default to daily prediction
def timeseries_L2_loss(pred_y, y, dt: float = 1.0):
    device = y.device
    time_integrate = dt * torch.ones(y.shape[1]).to(device)
    time_integrate[0] = 0.5 * dt
    time_integrate[-1] = 0.5 * dt
    loss_integral = ((pred_y - y) ** 2) @ time_integrate
    loss_value = torch.mean(loss_integral)
    return loss_value

def jL2_loss(pred_y, y, dt: float=1.0):
    time_integrate = dt * torch.ones(y.shape[-1])
    time_integrate[0] = 0.5 * dt
    time_integrate[-1] = 0.5 * dt
    loss_integral = lambda pred, true: ((pred - true) ** 2) @ time_integrate
    loss_values = eqx.filter_vmap(loss_integral)(pred_y, y)
    return jnp.mean(loss_values)


# H1 loss using C1 interpolants (use FD appx for gradient)
# default to daily prediction
def timeseries_H1_loss(pred_y, y, dt: float = 1.0):
    device = y.device
    time_integrate = dt * torch.ones(y.shape[1]).to(device)
    time_integrate[0] = 0.5 * dt
    time_integrate[-1] = 0.5 * dt

    L2_value = ((pred_y - y) ** 2) @ time_integrate
    H1_value = L2_value + (
        ((torch.gradient(pred_y - y, spacing=dt, edge_order=2)[0]) ** 2) @ time_integrate
    )

    loss_value = torch.mean(H1_value)
    return loss_value


def basis_ortho_loss(o_net_model, t, dt: float = 1.0, tol: float = 1e-3):
    time_integrate = dt * torch.ones_like(t)
    time_integrate[0] = 0.5 * dt
    time_integrate[-1] = 0.5 * dt
    weights = torch.diag(time_integrate.ravel())
    v = o_net_model.trunk_net(t)
    inner_products = v.T @ weights @ v
    v_norms = torch.sqrt(torch.diag(inner_products))
    v = v / v_norms
    inner_products = v.T @ weights @ v
    up_ids = np.triu_indices(inner_products.shape[0], 1)
    diag_ids = np.diag_indices(v.shape[1],2)
    basis_norm_small = torch.sum(F.relu(tol-torch.sqrt(inner_products[diag_ids]))**2)

    loss_value = torch.sum(inner_products[up_ids]**2) + basis_norm_small
    return loss_value

def jbasis_loss(o_net, t, dt: float = 1.0, tol: float = 1e-6):
    time_integrate = dt * jnp.ones_like(t)
    time_integrate[0] = 0.5 * dt
    time_integrate[-1] = 0.5 * dt
    weights = jnp.diag(time_integrate.ravel())
    v = o_net.trunk_net(t)
    inner_products = v.T @ weights @ v
    v_norms = jnp.sqrt(jnp.diag(inner_products))
    v = v / v_norms
    inner_products = v.T @ weights @ v
    up_ids = jnp.triu_indices(inner_products.shape[0], 1)
    diag_ids = jnp.diag_indices(v.shape[1],2)
    basis_norm_small = jnp.sum(jax.nn.relu(tol-jnp.sqrt(inner_products[diag_ids])))

    loss_value = jnp.sum(inner_products[up_ids]**2) + basis_norm_small
    return loss_value

def basis_non_ortho_norm(o_net_model, t, dt: float = 1.0):
    time_integrate = dt * torch.ones_like(t)
    time_integrate[0] = 0.5 * dt
    time_integrate[-1] = 0.5 * dt
    weights = torch.diag(time_integrate.ravel())
    v = o_net_model.trunk_net(t)
    inner_products = v.T @ weights @ v
    v_norms = torch.sqrt(torch.diag(inner_products))
    v = v / v_norms
    inner_products = v.T @ weights @ v
    up_ids = np.triu_indices(inner_products.shape[0], 1)
    # diag_ids = np.diag_indices(v.shape[1],2)

    loss_value = torch.norm(inner_products[up_ids])
    return loss_value

def jbasis_non_ortho_norm(o_net, t, dt: float = 1.0):
    time_integrate = dt * jnp.ones_like(t)
    time_integrate[0] = 0.5 * dt
    time_integrate[-1] = 0.5 * dt
    weights = jnp.diag(time_integrate.ravel())
    v = o_net.trunk_net(t)
    inner_products = v.T @ weights @ v
    v_norms = jnp.sqrt(jnp.diag(inner_products))
    v = v / v_norms
    inner_products = v.T @ weights @ v
    up_ids = jnp.triu_indices(inner_products.shape[0], 1)
    # diag_ids = np.diag_indices(v.shape[1],2)

    loss_value = jnp.linalg.norm(inner_products[up_ids])
    return loss_value


def out_of_range_loss(pred_y, min_val: float=0., max_val: float=1.):
    above = F.relu(pred_y - max_val)
    below = F.relu(min_val - pred_y)
    loss_value = torch.sum((above + below)**2)
    # limited = F.hardtanh(pred_y, min_val, max_val)
    # loss_value = torch.sum((pred_y-limited)**2)
    return loss_value

def jrange_loss(pred_y, min_val: float=0., max_val: float=1., dt: float=1.):
    time_integrate = dt * jnp.ones(pred_y.shape[-1])
    time_integrate[0] = 0.5 * dt
    time_integrate[-1] = 0.5 * dt
    above = jax.nn.relu(pred_y - max_val)
    below = jax.nn.relu(min_val - pred_y)
    loss_integral = lambda a, b: ((a+b) ** 2) @ time_integrate
    loss_values = eqx.filter_vmap(loss_integral)(above, below)
    return jnp.mean(loss_values)


def naive_mlp_epoch(
    model,
    optimizer,
    train_loader,
    loss_fn: callable = timeseries_MSE_loss,
    dt: float = 1.0,
    range_loss: float=0.1,
    cuda: bool = True
):
    model.train(True)
    running_loss = 0.0
    last_loss = 0.0

    for batch in iter(train_loader):
        xs, ys = batch
        if cuda:
            xs = xs.cuda()
            ys = ys.cuda()
        optimizer.zero_grad()

        pred_y = model(xs)

        loss = loss_fn(pred_y, ys, dt)
        if range_loss:
            loss = loss + out_of_range_loss(pred_y, 0, 1)
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
    noise_variance: float = 0.01,
    range_loss: float=0.1,
    cuda: bool = True
):
    model.train(True)
    running_loss = 0.0
    last_loss = 0.0

    for batch in iter(train_loader):
        xs, ys = batch
        noise = noise_variance * torch.randn(xs.shape)
        if cuda:
            xs = xs.cuda()
            ys = ys.cuda()
            noise = noise.cuda()
        optimizer.zero_grad()

        noisy_x = xs + noise
        pred_y = model(noisy_x)

        loss = loss_fn(pred_y, ys, dt)
        if range_loss:
            loss = loss + out_of_range_loss(pred_y, 0, 1)
        loss.backward()
        optimizer.step()

        last_loss = loss.item()
        running_loss += loss.item()

    return last_loss, running_loss / len(train_loader)


def noisy_onet_epoch(
    model,
    optimizer,
    train_loader,
    loss_fn: callable = timeseries_MSE_loss,
    dt: float = 1.0,
    noise_variance: float = 0.01,
    range_loss: float=0.1,
    ortho_loss: float=0.1,
    final_act: callable=None,
    cuda: bool = True,
):
    model.train(True)
    running_loss = 0.0
    last_loss = 0.0

    for batch in iter(train_loader):
        xs, ys = batch
        times = torch.linspace(0, 1, 285)[:,None]
        noise = noise_variance * torch.randn(xs.shape)
        if cuda:
            xs = xs.cuda()
            ys = ys.cuda()
            times = times.cuda()
            noise = noise.cuda()
        optimizer.zero_grad()
        
        noisy_x = xs + noise
        pred_y = model(noisy_x, times, final_act)

        loss = loss_fn(pred_y, ys, dt)
        if ortho_loss:
            loss = loss + ortho_loss*basis_ortho_loss(model, times, dt)
        if range_loss:
            loss = loss + range_loss*out_of_range_loss(pred_y, 0, 1)
        loss.backward()
        optimizer.step()

        last_loss = loss.item()
        running_loss += loss.item()

    return last_loss, running_loss / len(train_loader)

def j_onet_epoch(
    model,
    optimizer: opx.GradientTransformation,
    opt_state: opx.OptState,
    xs,
    ys,
    seed: int,
    loss_fn: callable = jL2_loss,
    dt: float = 1.0,
    noise_variance: float = 0.01,
    range_alpha: float=0.1,
    ortho_alpha: float=0.1,
    final_act: callable=None
):
    key = jrd.PRNGKey(seed)
    times = jnp.linspace(0,1,285)[:,None]
    def train_scan(carry, data):
        model, opt_state, run_loss, key = carry
        x, y = data
        keys = jrd.split(key,2)
        noise = noise_variance*jrd.normal(keys[0], x.shape)
        x = x+noise
        pred = model(x,times)

        misfit_loss, misfit_grads = eqx.filter_value_and_grad(loss_fn)(pred, y, dt)
        range_loss, range_grads = eqx.filter_value_and_grad(jrange_loss)(pred, y, dt)
        ortho_loss, ortho_grads = eqx.filter_value_and_grad(jbasis_loss)(pred, y, dt)

        run_loss = run_loss + misfit_loss + range_alpha*range_loss + ortho_alpha*ortho_loss
        grads = misfit_grads + range_alpha*range_grads + ortho_alpha*ortho_grads
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)

        return (model, opt_state, run_loss, keys[1]), None
    
    (model, opt_state, run_loss, _), _ = jax.lax.scan(train_scan, (model, opt_state, 0., key), (xs, ys))

    return model, opt_state, run_loss / xs.shape[0]





def mlp_test(
    model, test_loader, loss_fn: callable = timeseries_MSE_loss, dt: float = 1.0, cuda: bool = True
):
    running_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in iter(test_loader):
            xs, ys = batch
            if cuda:
                xs = xs.cuda()
                ys = ys.cuda()
            pred_y = model(xs)
            loss = loss_fn(pred_y, ys, dt)
            running_loss += loss.item()
            test_loss = running_loss / len(test_loader)
        return test_loss


def onet_test(
    model,
    test_loader,
    loss_fn: callable = timeseries_MSE_loss,
    dt: float = 1.0,
    final_act: callable=None,
    cuda: bool = True,
):
    running_loss = 0.0
    model.eval()
    times = torch.linspace(0, 1, 285)[:, None]
    with torch.no_grad():
        for batch in iter(test_loader):
            xs, ys = batch
            if cuda:
                xs = xs.cuda()
                ys = ys.cuda()
                times = times.cuda()
            pred_y = model(xs, times, final_act)
            loss = loss_fn(pred_y, ys, dt)
            running_loss += loss.item()
            test_loss = running_loss / len(test_loader)
        return test_loss


def j_onet_test(
    model,
    xs,
    ys,
    loss_fn: callable = jL2_loss,
    dt: float = 1.0,
):
    times = jnp.linspace(0,1,285)[:, None]
    def test_scan(carry, data):
        model, run_loss = carry
        x, y = data
        pred = model(x, times)
        loss = loss_fn(pred, y, dt)
        run_loss = run_loss + loss
        return (model, run_loss), None
    
    (_, run_loss), _ = jax.lax.scan(test_scan, (model, 0.), (xs, ys))

    return run_loss / xs.shape[0]



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
    dt: float = 1.0,
    final_act: callable=None,
    cuda: bool = True
):
    if log_wandb:
        wandb.init(project="DSML Final", name=name)

    filename = os.path.join(save_dir, name + ".pkl")
    if os.path.isfile(filename):
        os.rename(filename, filename + "-old")

    opt_filename = os.path.join(save_dir, name + "_opt.pkl")
    if os.path.isfile(opt_filename):
        os.rename(opt_filename, opt_filename + "-old")

    min_error = 1e5
    for epoch in range(num_epochs):
        last_loss, running_loss = naive_mlp_epoch(
            model, optimizer, train_loader, loss_fn, dt, final_act=final_act, cuda=cuda
        )
        test_loss = mlp_test(model, test_loader, loss_fn, final_act, cuda)

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

            with open(filename, "wb") as f:
                pkl.dump(model, f)

            with open(opt_filename, "wb") as f:
                pkl.dump(optimizer, f)


def noisy_train_mlp(
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
    range_loss: float=0.1,
    final_act: callable=None,
    cuda: bool = True,
):
    if log_wandb:
        wandb.init(project="DSML Final", name=name)

    min_error = 1e5
    for epoch in range(num_epochs):
        last_loss, running_loss = noisy_mlp_epoch(
            model, optimizer, train_loader, loss_fn, dt, noise_variance, range_loss, final_act=final_act, cuda=cuda
        )
        test_loss = mlp_test(model, test_loader, loss_fn, final_act, cuda)

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
                pkl.dump(model.state_dict(), f)

            opt_filename = os.path.join(save_dir, name + "_opt.pkl")
            if os.path.isfile(opt_filename):
                os.rename(opt_filename, opt_filename + "-old")
            with open(opt_filename, "wb") as f:
                pkl.dump(optimizer, f)


def noisy_train_onet(
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
    range_loss: float=0.1,
    ortho_loss: float=0.1,
    final_act: callable=None,
    cuda: bool = True,
):
    if log_wandb:
        wandb.init(project="DSML Final", name=name)
    
    min_error = 1e5
    times = torch.linspace[-1,1,285]
    for epoch in range(num_epochs):
        running_loss, last_loss = noisy_onet_epoch(
            model,
            optimizer,
            train_loader,
            loss_fn,
            dt,
            noise_variance,
            range_loss,
            ortho_loss,
            final_act=final_act,
            cuda=cuda,
        )
        test_loss = onet_test(model, test_loader, loss_fn, final_act=final_act, cuda=cuda)
        non_orthonormality = basis_non_ortho_norm(model, times, dt)
        if (epoch == num_epochs - 1) or ((epoch % print_every) == 0):
            if log_wandb:
                wandb.log(
                    {
                        "Epoch": epoch,
                        "Train Loss": running_loss,
                        "Test Loss": test_loss,
                        "Last Loss": last_loss,
                        "Basis Non-Orthonormality": non_orthonormality
                    }
                )
            print(
                "Epoch: {}   Train: {} ({})   Test: {}   Non-Ortho: {}".format(
                    epoch, running_loss, last_loss, test_loss, non_orthonormality
                )
            )

        if test_loss < min_error:
            min_error = test_loss
            filename = os.path.join(save_dir, name + ".pkl")
            if os.path.isfile(filename):
                os.rename(filename, filename + "-old")
            with open(filename, "wb") as f:
                pkl.dump(model.state_dict(), f)

            opt_filename = os.path.join(save_dir, name + "_opt.pkl")
            if os.path.isfile(opt_filename):
                os.rename(opt_filename, opt_filename + "-old")
            with open(opt_filename, "wb") as f:
                pkl.dump(optimizer, f)

def jtrain_o_net(
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
    range_alpha: float=0.1,
    ortho_alpha: float=0.1,
    final_act: callable=None
):
    if log_wandb:
        wandb.init(project="DSML Final", name=name)
    opt_state = optimizer.init(model)
    min_error = 1e5
    times = torch.linspace[-1,1,285]
    for epoch in range(num_epochs):
        xs = jnp.asarray([batch[0] for batch in iter(train_loader)])
        ys = jnp.asarray([batch[1] for batch in iter(train_loader)])
        model, opt_state, running_loss = j_onet_epoch(
            model,
            optimizer,
            opt_state,
            xs,
            ys,
            epoch,
            loss_fn,
            dt,
            noise_variance,
            range_alpha,
            ortho_alpha,
            final_act=final_act
        )
        test_loss = j_onet_test(model, test_loader, loss_fn, final_act=final_act)

        if (epoch == num_epochs - 1) or ((epoch % print_every) == 0):
            if log_wandb:
                wandb.log(
                    {
                        "Epoch": epoch,
                        "Train Loss": running_loss,
                        "Test Loss": test_loss
                    }
                )
            print(
                "Epoch: {}   Train: {} ({})   Test: {}".format(
                    epoch, running_loss, test_loss
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
                pkl.dump(opt_state, f)

    return model, opt_state

def noisy_train_MLPF(
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
    range_loss: float=0.1,
    final_act: callable=None,
    cuda: bool = True,
):
    if log_wandb:
        wandb.init(project="DSML Final", name=name)

    times = torch.linspace(0,1,285)[:,None]
    if cuda:
        times = times.cuda()
    min_error = 1e5
    for epoch in range(num_epochs):
        last_loss, running_loss = noisy_onet_epoch(
            model,
            optimizer,
            train_loader,
            loss_fn,
            dt,
            noise_variance,
            range_loss,
            ortho_loss=0,
            final_act=final_act,
            cuda=cuda,
        )
        test_loss = onet_test(model, test_loader, loss_fn, final_act=final_act, cuda=cuda)
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
                "Epoch: {}   Train: {} ({})   Test: {}".format(
                    epoch, running_loss, last_loss, test_loss
                )
            )

        if test_loss < min_error:
            min_error = test_loss
            filename = os.path.join(save_dir, name + ".pkl")
            if os.path.isfile(filename):
                os.rename(filename, filename + "-old")
            with open(filename, "wb") as f:
                pkl.dump(model.state_dict(), f)

            opt_filename = os.path.join(save_dir, name + "_opt.pkl")
            if os.path.isfile(opt_filename):
                os.rename(opt_filename, opt_filename + "-old")
            with open(opt_filename, "wb") as f:
                pkl.dump(optimizer, f)
