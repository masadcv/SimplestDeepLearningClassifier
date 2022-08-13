import torch


def get_optimiser(optimiser_name, params, lr):
    optimiser_dict = {
        "SGD": torch.optim.SGD,
        "ADAM": torch.optim.Adam,
        "ADADETLA": torch.optim.Adadelta,
        "RMSPROP": torch.optim.RMSprop,
    }

    return optimiser_dict[optimiser_name](params, lr=lr)
