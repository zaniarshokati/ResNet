import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
from livelossplot import PlotLosses


class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=False,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience
        self.best_epoch = None

        self._mean_f1_score_epoch = None

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):

        # perform following steps:
        # -reset the gradients
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        # TODO
        self._optim.zero_grad()
        x_tensor = t.tensor(x, dtype=t.float).clone().detach().requires_grad_(True)
        y_tensor = t.tensor(y, dtype=t.float).clone().detach().requires_grad_(True)
        y_predict = self._model.forward(x_tensor)
        loss = self._crit(y_predict, y_tensor)
        loss.backward()
        self._optim.step()
        return loss

    def val_test_step(self, x, y):
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        # TODO
        y_predict = self._model.forward(x)
        loss = self._crit(y_predict, y)
        return loss, y_predict

    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        # TODO
        epoch_loss = 0
        num_batches = 0
        for image, label in self._train_dl:
            num_batches += 1
            if self._cuda:
                image = image.cuda()
                label = label.cuda()
            epoch_loss += self.train_step(image, label)

        mean_loss = epoch_loss / num_batches
        print('Mean training loss for this epoch computed ' + str(mean_loss))
        return mean_loss

    def val_test(self):
        # set eval mode
        # disable gradient computation
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        # TODO
        self._val_test_dl.dataset.mode = 'val'
        epoch_loss = 0
        f1_score_epoch = 0
        num_batches = 0
        with t.no_grad():
            for image, label in self._val_test_dl:
                num_batches += 1
                if self._cuda:
                    image = image.cuda()
                    label = label.cuda()
                loss, y_predict = self.val_test_step(image, label)
                epoch_loss += loss
                f1_score_epoch += f1_score(y_predict.cpu().numpy() > 0.5, label.cpu().numpy(),
                                           average='micro')  # TODO change this
            mean_f1_score_epoch = f1_score_epoch / num_batches
            print('Mean f1_score for the epoch: ' + str(mean_f1_score_epoch))
            self._mean_f1_score_epoch = mean_f1_score_epoch

        mean_loss_epoch = epoch_loss / num_batches
        return mean_loss_epoch

    def fit(self, epochs=-1):

        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        # TODO
        counter_epoch = 0
        loss_training = []
        loss_validation = []
        counter_to_stop = 0
        prev_loss = 1E2
        best_loss_validation = 1E2
        liveloss = PlotLosses()
        mean_f1_score_epochs = []
        best_mean_f1_score_epoch = 0
        while counter_to_stop < self._early_stopping_patience and counter_epoch < epochs:

            logs = {}
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
            # TODO
            counter_epoch += 1
            print('epoch: ' + str(counter_epoch))
            loss_training.append(self.train_epoch())
            loss_validation.append(self.val_test())
            print('validation loss: ' + str(loss_validation[-1]))

            if prev_loss - 1E-3 < loss_validation[-1]:
                counter_to_stop += 1
            else:
                counter_to_stop = 0
            prev_loss = loss_validation[-1]
            mean_f1_score_epoch = t.tensor(self._mean_f1_score_epoch, device='cpu')
            mean_f1_score_epochs.append(mean_f1_score_epoch)
            if mean_f1_score_epochs[-1] > best_mean_f1_score_epoch:  # loss_validation[-1] < best_loss_validation:
                # best_loss_validation = loss_validation[-1]
                best_mean_f1_score_epoch = mean_f1_score_epoch
                self.best_epoch = counter_epoch
                self.save_checkpoint(counter_epoch)

            logs['val_' + 'f1_score_mean'] = mean_f1_score_epoch
            logs['' + 'f1_score_mean'] = 0.85
            logs['' + 'log loss'] = t.tensor(loss_training[-1], device='cpu')
            logs['val_' + 'log loss'] = t.tensor(prev_loss, device='cpu')
            liveloss.update(logs)
            liveloss.send()

        return loss_training, loss_validation
