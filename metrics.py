import sklearn.metrics as metrics
import numpy as np
import pandas as pd


def print_metrics(target, output, num_classes):
    print('metrics')
    acc_dict = {}
    for i in range(num_classes):
        print('Class:', i)
        labels = np.sort(np.unique(target[:, i])).astype(int)
        cm = metrics.confusion_matrix(target[:, i], output[:, i])
        cm_df = pd.DataFrame(cm, columns=labels, index=labels)
        acc = metrics.accuracy_score(target[:, i], output[:, i])

        acc_tag = str(i) + "_Accuracy"
        acc_dict[acc_tag] = acc
        class_cm_dict = {}
        for j in labels:
            tag = str(i) + "_" + str(j) + "_tp"
            if sum(cm_df.loc[j]) > 0:
                class_cm_dict[tag] = cm_df.loc[j][j]/sum(cm_df.loc[j])
        wandb.log(class_cm_dict)
    wandb.log(acc_dict)
