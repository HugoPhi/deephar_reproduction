import src.config as config
import src.task as task
import numpy as np
import os
from itertools import product

class HyperSearch():
    def __init__(self, lab_name, model_name, learning_rate, dropout_rate, lambda_l2, batch_size, epochs):
        self.lab_name = lab_name
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.lambda_l2 = lambda_l2
        self.batch_size = batch_size
        self.epochs = epochs
        self.recorder = {}

        if not os.path.exists(f"./{self.lab_name}"):
            os.makedirs(f"./{self.lab_name}")
        self.file = open(f"./{self.lab_name}/log.txt", 'a')

    def __del__(self):
        self.file.close()

    def grid(self, repeat=5):
        best = {}
        self.file.write(f"grid search\n")
        self.file.write(f"raws:\n")
        for lr, dr, l2, bs, ep in product(self.learning_rate, self.dropout_rate, self.lambda_l2, self.batch_size, self.epochs):
            id = (lr, dr, l2, bs, ep)
            conf = config.Config(
                model = self.model_name,
                input_signal_types=['body_acc_x', 'body_acc_y', 'body_acc_z', 'body_gyro_x', 'body_gyro_y', 'body_gyro_z', 'total_acc_x', 'total_acc_y', 'total_acc_z'],
                labels=['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING'],
                dataset_path = 'data/UCI/UCI HAR Dataset/',
                time_steps = 128,
                hidden = 32,
                batch_size = bs,
                epochs = ep,
                learning_rate = lr,
                dropout_rate = dr,
                lambda_l2 = l2
            )

            t = task.Task(conf)
            p1 = t.train()
            p2 = t.evaluate(p1)
            self.recorder[id] = (p2['precision_score'], p2['recall_score'], p2['f1_score'])
            for _ in range(repeat - 1):  # repeat the same experiment for 5 times 
                tt = task.Task(conf)
                pp1 = t.train()
                pp2 = t.evaluate(pp1)
                self.recorder[id] = (self.recorder[id][0] + pp2['precision_score'], self.recorder[id][1] + pp2['recall_score'], self.recorder[id][2] + pp2['f1_score'])

            # get average
            self.recorder[id] = (self.recorder[id][0] / repeat, self.recorder[id][1] / repeat, self.recorder[id][2] / repeat)
            self.file.write(f"> {id}: {self.recorder[id]}\n")
            t.figure(p2, plot = False)

        self.file.write(f"fused:\n")
        fused = {id : prc + rec + f1 for id, (prc, rec, f1) in self.recorder.items()}
        fused = sorted(fused.items(), key=lambda x: x[1])
        for id, prc in fused:
            self.file.write(f'> {id}: {prc}\n')

        best = [id for id, prc in fused if prc == fused[-1][1]]
        self.file.write(f'best: {best}')
        return best

if __name__ == "__main__":
    # hp = HyperSearch(
    #     lab_name = 'lab1',
    #     model_name = 'res_bi_lstm',
    #     learning_rate = [0.001, 0.0015, 0.0025],
    #     dropout_rate = [0.5, 0.85],
    #     lambda_l2 = [0.001, 0.0015, 0.0025, 0.005],
    #     lambda_l2 = [0.001, 0.0015, 0.0025, 0.005],
    #     batch_size = [256, 512, 1024, 2048], 
    #     epochs = [1]
    # )
    hp = HyperSearch(
        lab_name = 'lab1',
        model_name = 'res_bi_lstm',
        learning_rate = [0.001, 0.0025],
        dropout_rate = [0.5],
        lambda_l2 = [0.0015],
        batch_size = [512], 
        epochs = [1, 5]
    )
    best = hp.grid()
    print(f'best is {best}')
