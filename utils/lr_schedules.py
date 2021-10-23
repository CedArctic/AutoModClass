from math import cos, pi, exp, floor

warmup_rate = 0.1


def szegedy_schedule(epoch, lr):
    if epoch == 1:
        pass
    elif epoch % 2 == 1:
        lr = 0.7 * lr
    return lr


def cosine_annealing_schedule(epoch, lr):
    epochs_total = 30
    lr = (1 / 2) * (1 + cos(epoch * pi / epochs_total)) * lr
    return lr


def exp_decay_schedule(epoch, lr):
    initial_lr = 0.001
    return initial_lr * exp(-0.1 * epoch)


# Also known as Learning Rate Annealing
def step_decay_schedule(epoch, lr):
    # Number of epochs after which the learning rate should drop
    epochs_drop = 10
    # Constant rate at which the drop happens
    drop = 0.1
    # Initial Learning Rate
    initial_lr = 0.001

    lr = initial_lr * (drop ** floor(epoch / epochs_drop))
    return lr

# TODO:ALSO TRY REVERSE HERE
def triangular_cyclic_schedule(epoch, lr):
    # Minimum and Maximum Learning Rates
    lr_min = 0.00001
    lr_max = 0.0001

    # Number of epochs for a cycle
    stepsize = 10

    cycle = floor(1 + epoch / (2 * stepsize))
    x = abs(epoch / stepsize - 2 * cycle + 1)
    lr = lr_min + (lr_max - lr_min) * max(0, 1 - x)

    return lr

