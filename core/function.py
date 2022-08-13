import os
import torch

import utils


def train(args, model, train_loader, optimizer, loss_func, epoch):
    losses = utils.AverageMeter()

    # run in train mode
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        # move data to device
        data, target = (
            data.to(args["device"]),
            target.to(args["device"]),
        )
        # zero grad
        optimizer.zero_grad()

        # forward pass
        output = model(data)
        loss = loss_func(output, target)

        # backward pass
        loss.backward()
        losses.update(loss.item(), data.size(0))

        # update parameters
        optimizer.step()

        # print loss at defined interval
        if batch_idx % args["log_interval"] == 0 and batch_idx != 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * train_loader.batch_size,
                    len(train_loader) * train_loader.batch_size,
                    100.0 * batch_idx / len(train_loader),
                    losses.avg,
                )
            )

    # write train loss curve to tensorboard for monitoring
    if "tb_writer" in args.keys():
        args["tb_writer"].add_scalar(
            "{}_loss".format(args["epoch_name"]), losses.avg, epoch
        )


def evaluate(args, model, test_loader, loss_func, epoch):
    losses = utils.AverageMeter()
    working_dir = args["tb_writer"].log_dir if "tb_writer" in args.keys() else "/tmp"
    conf_file = os.path.join(
        working_dir, "confplot_{}_{:05d}.png".format(args["epoch_name"], epoch)
    )
    confplotter = utils.ConfMatrixPlotter(args["conf_matrix_func"], save_path=conf_file)

    # run in eval mode
    model.eval()

    with torch.no_grad():
        for data, target in test_loader:

            # move data to device
            data, target = (
                data.to(args["device"]),
                target.to(args["device"]),
            )

            # forward pass
            output = model(data)
            test_loss = loss_func(output, target)

            losses.update(test_loss.item(), data.size(0))
            pred = torch.argmax(output, dim=1)

            confplotter.update(pred, target)

            # collect metrics from predictions
            for vm in args["val_metrics"].keys():
                args["val_metrics"][vm].update(pred, target)

    # print accuracy from metrics
    accuracy = {}
    for vm in args["val_metrics"].keys():
        accuracy[vm] = args["val_metrics"][vm].compute()
        # reset current metric
        args["val_metrics"][vm].reset()

    accuracy_str = ""
    for vm in accuracy.keys():
        # compute metric and add to accuracy string
        accuracy_str += "{}: {:.4f}  ".format(vm, accuracy[vm])

    print(
        "\n{} set: Average loss: {:.4f}\nAccuracy -> {:s}\n".format(
            args["epoch_name"],
            losses.avg,
            accuracy_str,
        )
    )

    # write logs to tensorboard
    if "tb_writer" in args.keys():
        args["tb_writer"].add_scalar(
            "{}_loss".format(args["epoch_name"]), losses.avg, epoch
        )
        for vm in accuracy.keys():
            args["tb_writer"].add_scalar(
                "{}_acc_{}".format(args["epoch_name"], vm),
                accuracy[vm],
                epoch,
            )

        # write conf matrix plots for pred and target
        figure = confplotter.plot_confusion_matrix()
        args["tb_writer"].add_figure(
            "{}_conf_matrix".format(args["epoch_name"]), figure, epoch
        )
        figure.clf()

    return accuracy, losses.avg


def evaluate_with_results(args, model, test_loader, loss_func, epoch):
    losses = utils.AverageMeter()

    # run in eval mode
    model.eval()

    # result gatherers to collect/return all predictions
    out_result = utils.ResultGatherer()

    with torch.no_grad():
        for data, target in test_loader:

            # move data to device
            data, target = (
                data.to(args["device"]),
                target.to(args["device"]),
            )

            # forward pass
            output = model(data)

            test_loss = loss_func(output, target)

            losses.update(test_loss.item(), data.size(0))
            pred = torch.argmax(output, dim=1)

            out_result.update(pred, target)

            for vm in args["val_metrics"].keys():
                args["val_metrics"][vm].update(pred, target)

    # print accuracy
    accuracy = {}
    for vm in args["val_metrics"].keys():
        accuracy[vm] = args["val_metrics"][vm].compute().detach().cpu().numpy().item()
        # reset current metric
        args["val_metrics"][vm].reset()

    accuracy_str = ""
    for vm in accuracy.keys():
        # compute metric and add to accuracy string
        accuracy_str += "{}: {:.4f}  ".format(vm, accuracy[vm])

    print(
        "\nAverage loss: {:.4f}\nAccuracy -> {:s}\n".format(
            losses.avg,
            accuracy_str,
        )
    )

    # get each pred/target from gatherers
    out_result.finalise()

    # return overall accuracy and individual results
    return (
        accuracy,
        losses.avg,
        out_result.output,
        out_result.target,
    )
