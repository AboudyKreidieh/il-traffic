"""TODO."""
import numpy as np
import torch


def get_flat_grads(f, net):
    """TODO.

    Parameters
    ----------
    f : TODO
        TODO
    net : TODO
        TODO

    Returns
    -------
    TODO
        TODO
    """
    flat_grads = torch.cat([
        grad.view(-1)
        for grad in torch.autograd.grad(f, net.parameters(), create_graph=True)
    ])

    return flat_grads


def get_flat_params(net):
    """TODO.

    Parameters
    ----------
    net : TODO
        TODO

    Returns
    -------
    TODO
        TODO
    """
    return torch.cat([param.view(-1) for param in net.parameters()])


def set_params(net, new_flat_params):
    """TODO.

    Parameters
    ----------
    net : TODO
        TODO
    new_flat_params : TODO
        TODO

    Returns
    -------
    TODO
        TODO
    """
    start_idx = 0
    for param in net.parameters():
        end_idx = start_idx + np.prod(list(param.shape))
        param.data = torch.reshape(
            new_flat_params[start_idx:end_idx], param.shape
        )

        start_idx = end_idx


def conjugate_gradient(Av_func, b, max_iter=10, residual_tol=1e-10):
    """TODO.

    Parameters
    ----------
    Av_func : TODO
        TODO
    b : TODO
        TODO
    max_iter : TODO
        TODO
    residual_tol : TODO
        TODO

    Returns
    -------
    TODO
        TODO
    """
    x = torch.zeros_like(b)
    r = b - Av_func(x)
    p = r
    rsold = r.norm() ** 2

    for _ in range(max_iter):
        Ap = Av_func(p)
        alpha = rsold / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.norm() ** 2
        if torch.sqrt(rsnew) < residual_tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


def rescale_and_linesearch(g, s, Hs, max_kl, L, kld, old_params, pi,
                           max_iter=10, success_ratio=0.1):
    """TODO.

    Parameters
    ----------
    g : TODO
        TODO
    s : TODO
        TODO
    Hs : TODO
        TODO
    max_kl : TODO
        TODO
    L : TODO
        TODO
    kld : TODO
        TODO
    old_params : TODO
        TODO
    pi : TODO
        TODO
    max_iter : TODO
        TODO
    success_ratio : TODO
        TODO

    Returns
    -------
    TODO
        TODO
    """
    set_params(pi, old_params)
    L_old = L().detach()

    beta = torch.sqrt((2 * max_kl) / torch.dot(s, Hs))

    for _ in range(max_iter):
        new_params = old_params + beta * s

        set_params(pi, new_params)
        kld_new = kld().detach()

        L_new = L().detach()

        actual_improv = L_new - L_old
        approx_improv = torch.dot(g, beta * s)
        ratio = actual_improv / approx_improv

        if ratio > success_ratio and actual_improv > 0 and kld_new < max_kl:
            return new_params

        beta *= 0.5

    print("The line search was failed!")
    return old_params
