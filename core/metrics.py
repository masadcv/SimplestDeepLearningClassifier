import torchmetrics


def get_metric(metric_name, num_classes=10, average="micro"):
    metricname_to_func = {
        "Accuracy": torchmetrics.Accuracy(num_classes=num_classes, average=average),
        "AUROC": torchmetrics.AUROC(num_classes=num_classes, average=average),
        "CONF": torchmetrics.ConfusionMatrix(num_classes=num_classes, normalize="true"),
        "F1": torchmetrics.F1Score(num_classes=num_classes, average=average),
        "Precision": torchmetrics.Precision(num_classes=num_classes, average=average),
        "Recall": torchmetrics.Recall(num_classes=num_classes, average=average),
        "Specificity": torchmetrics.Specificity(
            num_classes=num_classes, average=average
        ),
    }
    return metricname_to_func[metric_name]
