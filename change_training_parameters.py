import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd

# parameters to change
_epochs = [40, 60, 80, 100]
_BatchSize = [16, 24, 32, 68]
_learningRate = [1.0e-4, 2.4e-4, 4.8e-4, 9.6e-4]
_decay_weight = [1e-5, 5e-6, 2.5e-6, 1.25e-6]
f = open("myfile.txt", "a")
for e in _epochs:
    for bs in _BatchSize:
        for lr_ in _learningRate:
            for dw in _decay_weight:
                csv_path = 'data.csv'
                tab = pd.read_csv(csv_path, sep=';')
                tab_train = tab.iloc[500:, :].reset_index()
                tab_val = tab.iloc[:500, :].reset_index()

                train_dl = t.utils.data.DataLoader(ChallengeDataset(tab_train, 'train'), batch_size=bs, shuffle=True)
                val_dl = t.utils.data.DataLoader(ChallengeDataset(tab_val, 'val'), batch_size=bs, shuffle=True)

                num_epochs = _epochs
                learningRate = _learningRate
                decay_weight = _decay_weight

                myModel = model.ResNet()
                crit = t.nn.BCELoss()
                optim = t.optim.Adam(myModel.parameters(), lr=lr_, weight_decay=decay_weight)
                trainer = Trainer(myModel, crit=crit, optim=optim, cuda=False, train_dl=train_dl, val_test_dl=val_dl,
                                  early_stopping_patience=10)
                res = trainer.fit(num_epochs)
                res = t.tensor(res, device='cpu')
                trainer.save_onnx('checkpoint_{:03d}.onnx'.format(trainer.best_epoch))

                # plot the results
                plt.figure()
                plt.plot(np.arange(len(res[0])), res[0], label='Loss: Training')
                plt.plot(np.arange(len(res[1])), res[1], label='Loss: Validation')
                plt.yscale('log')
                plt.legend()
                plt.savefig("e:" + str(e) +"BS:" + str(bs) + "LR:" + str(lr_) + "DW:" + str(dw)+".png")
                plt.show()
                f.write("/n epoch:" + str(e) + "   " + "BatchSize:" + str(bs) + "   " + "LearningRate:" + str(lr_) + "   " + "DecayWeight:" + str(dw))
f.close()

