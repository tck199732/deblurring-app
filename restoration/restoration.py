import numpy as np


def total_variation(X, lambda_tv):
    # implement TV regularization, image is not considered circular for now
    reg_factor = np.ones_like(X)
    for i in range(1, len(X) - 1):
        if (X[i] < X[i-1]) and (X[i] < X[i+1]):
            reg_factor[i] = 1. / (1 - lambda_tv)
        elif (X[i] > X[i-1]) and (X[i] > X[i+1]):
            reg_factor[i] = 1. / (1 + lambda_tv)
    return reg_factor

def richardson_lucy(X, y, psf, **kwargs):
    """ Richardson-Lucy Algorithm for 1D Image Deconvolution
    Parameters
    ----------
    X : array_like
        Initial guess of the underlying image (shape (M,))
    y : array_like
        observed image (shape (N,))
    psf : array_like
        kernel matrix / point spread function in the forward model (shape (N, M))
    """    
    reg = kwargs.pop("reg", "std")
    if not reg in ['std', 'tv', 'mem']:
        raise ValueError("Invalid regularization method '%s'" % reg)

    methods = {
        'std': rl_standard,
        'tv': rl_total_variation,
        'mem': rl_maximum_entropy
    }
    return methods[reg](X, y, psf, **kwargs)

def rl_standard(X, y, psf, **kwargs):
    # handle keyword arguments
    accel_rate = kwargs.pop("accel_rate", 1.99)      # acceleration rate
    niter = kwargs.pop("niter", 50000)              # maximum number of iterations
    epsilon = kwargs.pop("epsilon", 1e-12)          # small value to avoid division by zero
    tol = kwargs.pop("tol", 1e-3)                   # tolerance for convergence

    if len(kwargs) > 0:
        raise TypeError(
            "richardson_lucy_me() got unexpected keyword arguments '%s'" % ', '.join(list(kwargs.keys()))
        )
    
    # handle normalization of inputs
    y_norm = np.sum(y)
    psf_norm = np.sum(psf, axis=0)

    psf_ = psf / psf_norm
    img = y / y_norm
    deconv = X / np.sum(X)
    conv = np.dot(psf_, deconv) + epsilon

    X_preds, y_preds = [], []
    for _ in range(niter):
        deconv = deconv * (np.einsum('i,ij', img / conv, psf_)) ** accel_rate
        conv = np.dot(psf_, deconv) + epsilon

        X_preds.append(deconv * y_norm / psf_norm)
        y_preds.append(conv * y_norm)

    return X_preds, y_preds


def rl_total_variation(X, y, psf, **kwargs):
    # handle keyword arguments
    tol = kwargs.pop("tol", 1e-3)                   # tolerance for convergence
    lambda_tv = kwargs.pop("lambda_tv", 0.005)        # total variation regularization strength
    renorm = kwargs.pop("renorm", True)             # flag to re-normalize the image after each iteration
    
    accel_rate = kwargs.pop("accel_rate", 1.99)      # acceleration rate
    niter = kwargs.pop("niter", 50000)              # maximum number of iterations
    epsilon = kwargs.pop("epsilon", 1e-12)          # small value to avoid division by zero

    if len(kwargs) > 0:
        raise TypeError(
            "richardson_lucy_me() got unexpected keyword arguments '%s'" % ', '.join(list(kwargs.keys()))
        )

    # handle normalization of inputs
    y_norm = np.sum(y)
    psf_norm = np.sum(psf, axis=0)

    psf_ = psf / psf_norm
    img = y / y_norm
    deconv = X / np.sum(X)
    conv = np.dot(psf_, deconv) + epsilon

    X_preds, y_preds = [], []
    for _ in range(niter):

        reg_factor = total_variation(deconv, lambda_tv)
        deconv = deconv * (np.einsum('i,ij', img / conv, psf_)) ** accel_rate * reg_factor

        # explicitly re-normalize
        if lambda_tv > 0:
            deconv /= np.sum(deconv)

        conv = np.dot(psf_, deconv) + epsilon

        X_preds.append(deconv * y_norm / psf_norm)
        y_preds.append(conv * y_norm)
        
    return X_preds, y_preds

def rl_maximum_entropy(X, y, psf, **kwargs):

    # handle keyword arguments
    alpha = kwargs.pop("alpha", 0.0)                # regularization strength in mem
    tol_norm = kwargs.pop("tol_norm", 1e-6)         # tolerance for image normalization in mem
    tol = kwargs.pop("tol", 1e-3)                   # tolerance for convergence
    Pi = kwargs.pop("Pi", None)                     # smoothing matrix for regularizing the restored image

    assert Pi is not None, "Smoothing matrix Pi is required for Maximum Entropy Method"
    # common options
    accel_rate = kwargs.pop("accel_rate", 1.99)      # acceleration rate
    accelerated = (accel_rate > 1.0)
    niter = kwargs.pop("niter", 50000)              # maximum number of iterations
    epsilon = kwargs.pop("epsilon", 1e-12)          # small value to avoid division by zero

    if len(kwargs) > 0:
        raise TypeError(
            "richardson_lucy_me() got unexpected keyword arguments '%s'" % ', '.join(list(kwargs.keys()))
        )

    # handle normalization of inputs
    y_norm = np.sum(y)
    psf_norm = np.sum(psf, axis=0)

    psf_ = psf / psf_norm
    img = y / y_norm
    Pi_ = Pi / np.sum(Pi, axis=0)

    deconv = X / np.sum(X)
    conv = np.dot(psf_, deconv) + epsilon

    X_preds, y_preds, ts = [], [], []

    for _ in range(niter):
        deltaH = deconv * (np.einsum('i,ij', img / conv, psf_) - 1)
        # conservation of flux
        assert np.abs(np.sum(deconv) - 1) < tol_norm
        assert np.abs(np.sum(deltaH)) < tol_norm

        deltaS = np.zeros_like(deconv)
        if alpha > 0:
            chi = np.dot(Pi_, deconv)
            Sratio = deconv / chi
            lnSratio = np.log(Sratio)
            S = - np.sum(deconv * lnSratio)
            deltaS = -alpha * deconv * (
                1 + lnSratio + S - np.dot(Pi_.T, Sratio)
            )
            assert np.abs(np.sum(deltaS)) < tol_norm

        delta = deltaH + deltaS

        accel = 1.0
        if accelerated:
            # from positivity constraint
            accel_c = np.divide(deconv, np.abs(delta), out=np.zeros_like(deconv), where=delta!=0)
            accel_c = accel_c[delta < 0].min()
            accel = min(accel_c, accel_rate)
            assert accel > 1

        deconv = deconv + accel * delta
        conv = np.dot(psf_, deconv) + epsilon
        t = np.abs(delta) / (np.abs(deltaH) + np.abs(deltaS))

        X_preds.append(deconv * y_norm / psf_norm)
        y_preds.append(conv * y_norm)
        ts.append(t)

        if alpha > 0. and np.all(t < tol):
            break
    
    return X_preds, y_preds, ts
