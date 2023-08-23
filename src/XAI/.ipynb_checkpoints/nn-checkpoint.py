import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ProgressBar

# Define a PyTorch Lightning Module
class BinaryClassifier(pl.LightningModule):
    def __init__(self, mean, std, num_cols):
        super(BinaryClassifier, self).__init__()
        self.scaler = CustomScalingLayer(mean, std)
        self.layer_1 = nn.Linear(num_cols, 64)
        self.layer_2 = nn.Linear(64, 32)
        self.layer_out = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(32)
        
    def forward(self, inputs):
        x = self.scaler(inputs)
        x = x.to(self.layer_1.weight.dtype) 
        x = self.relu(self.layer_1(x))
        x = self.batchnorm1(x)
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.BCEWithLogitsLoss()(y_pred.view(-1), y.type_as(y_pred))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        y_pred_tag = (torch.sigmoid(y_pred.view(-1)) > 0.5).float()
        correct = (y_pred_tag == y).float().sum()
        acc = correct/y.shape[0]
        self.log('val_acc', acc)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


class CustomScalingLayer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.nn.Parameter(mean, requires_grad=False)
        self.std = torch.nn.Parameter(std, requires_grad=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class SilentProgressBar(ProgressBar):

    def disable(self):
        self.enabled = False


def get_predictions(model, dataloader):
    model.eval()  # set the model to evaluation mode
    predictions = []
    with torch.no_grad():  # deactivate autograd engine to reduce memory usage and speed up computations
        for batch in dataloader:
            data, _ = batch
            outputs = model(data)
            predicted = (torch.sigmoid(outputs.view(-1)) > 0.5).float()
            predictions.extend(predicted.tolist())
    return predictions