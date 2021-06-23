from math import cos, pi, exp

warmup_rate = 0.1


def szegedy_schedule(epoch, lr):
    if epoch == 1:
        pass
    elif epoch % 2 == 1:
        lr = 0.94 * lr
    return lr


def cosine_annealing_schedule(epoch, lr):
    epochs_total = 30
    lr = (1/2) * (1 + cos(epoch * pi / epochs_total)) * lr
    return lr


def exp_decay_schedule(epoch, lr):
    initial_lr = 0.001
    return initial_lr * exp(-0.1 * epoch)
