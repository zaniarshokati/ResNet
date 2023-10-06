import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
from livelossplot import PlotLosses


class Trainer:
    def __init__(
        self,
        model,  # Model to be trained.
        crit,  # Loss function
        optim=None,  # Optimizer
        train_dl=None,  # Training data set
        val_test_dl=None,  # Validation (or test) data set
        cuda=False,  # Whether to use the GPU
        early_stopping_patience=-1,
    ):  # The patience for early stopping
        """
        Initializes the Trainer class.

        Args:
            model (torch.nn.Module): The model to be trained.
            crit (torch.nn.Module): The loss function.
            optim (torch.optim.Optimizer): The optimizer.
            train_dl (torch.utils.data.DataLoader): Training data loader.
            val_test_dl (torch.utils.data.DataLoader): Validation or test data loader.
            cuda (bool): Whether to use GPU (default is False).
            early_stopping_patience (int): The patience for early stopping (default is -1, which disables early stopping).
        """
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

    def train_step(self, x, y):
        """
        Performs a single training step.

        Args:
            x (torch.Tensor): Input data.
            y (torch.Tensor): Target data.

        Returns:
            torch.Tensor: Loss for the current step.
        """
        self._optim.zero_grad()
        x_tensor = t.tensor(x, dtype=t.float).clone().detach().requires_grad_(True)
        y_tensor = t.tensor(y, dtype=t.float).clone().detach().requires_grad_(True)
        y_predict = self._model.forward(x_tensor)
        loss = self._crit(y_predict, y_tensor)
        loss.backward()
        self._optim.step()
        return loss

    def fit(self, epochs=-1):
        """
        Trains the model for a specified number of epochs.

        Args:
            epochs (int): The number of epochs to train for (-1 for early stopping only).

        Returns:
            list: Training losses.
            list: Validation losses.
        """
        assert self._early_stopping_patience > 0 or epochs > 0
        counter_epoch = 0
        loss_training = []
        loss_validation = []
        counter_to_stop = 0
        prev_loss = 1e2
        best_loss_validation = 1e2
        liveloss = PlotLosses()
        mean_f1_score_epochs = []
        best_mean_f1_score_epoch = 0
        while (
            counter_to_stop < self._early_stopping_patience and counter_epoch < epochs
        ):
            logs = {}
            counter_epoch += 1
            print("epoch: " + str(counter_epoch))
            loss_training.append(self.train_epoch())
            loss_validation.append(self.val_test())
            print("validation loss: " + str(loss_validation[-1]))

            if prev_loss - 1e-3 < loss_validation[-1]:
                counter_to_stop += 1
            else:
                counter_to_stop = 0
            prev_loss = loss_validation[-1]
            mean_f1_score_epoch = t.tensor(self._mean_f1_score_epoch, device="cpu")
            mean_f1_score_epochs.append(mean_f1_score_epoch)
            if mean_f1_score_epochs[-1] > best_mean_f1_score_epoch:
                best_mean_f1_score_epoch = mean_f1_score_epoch
                self.best_epoch = counter_epoch
                self.save_checkpoint(counter_epoch)

            logs["val_" + "f1_score_mean"] = mean_f1_score_epoch
            logs["" + "f1_score_mean"] = 0.85
            logs["" + "log loss"] = t.tensor(loss_training[-1], device="cpu")
            logs["val_" + "log loss"] = t.tensor(prev_loss, device="cpu")
            liveloss.update(logs)
            liveloss.send()

        return loss_training, loss_validation
