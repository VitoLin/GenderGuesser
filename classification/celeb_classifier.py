import sys
sys.path.append('..')

import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.utils.data as torchdata
import pytorch_lightning as pt
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path
import argparse

from src.data_loader import EmbeddingData

class MLPClassifier(nn.Module):
    '''
    Input: 512
    ReLU
    Hidden1: 1000
    ReLU
    Hidden2: 1000
    ReLU
    Out: 1
    '''
    def __init__(self, input_features = 512):
        super().__init__()

        self.input = nn.Linear(input_features, 1000)
        self.hidden1 = nn.Linear(1000, 1000)
        self.hidden2 = nn.Linear(1000, 1000)
        self.out = nn.Linear(1000, 1)
    
    def forward(self, x):
        
        x = relu(self.input(x))
        x = relu(self.hidden1(x))
        x = relu(self.hidden2(x))
        x = torch.sigmoid(self.out(x)) # to convert to 0, 1
        return x

class MLPClassifierWrapper(pt.LightningModule):
    
    
    def __init__(self, input_features = 512):
        super().__init__()
        self.trained_model = None
        self.model = MLPClassifier(input_features=input_features)
        self.loss_func = nn.BCELoss()
        self.lr = .01
    
    
    def training_step(self, batch, batch_idx):
        # training loop
        x, y = batch
        yhat = self.model(x)
        loss = self.loss_func(yhat, y.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # validation loop
        x, y = batch
        yhat = self.model(x)
        loss = self.loss_func(yhat, y.float())
        val_acc = torch.sum(yhat.round() == y) / len(y)
        self.log("val_loss", loss)
        self.log("val_acc", val_acc)
        return loss
    
    def test_step(self, batch, batch_idx):
        # Testing
        x,y = batch
        yhat = self.model(x)
        test_acc = torch.sum(yhat.round() == y) / len(y)
        self.log("test_acc", test_acc)
        return test_acc
    
    def forward(self, x):
        ''' for real time later.'''
        # self.trained_model must be set first
        if self.trained_model:
            x = self.trained_model(x)
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)# weight_decay=.00005)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode = 'min', factor = .1, patience=4, min_lr=1e-8, verbose = True),
            "monitor": "val_loss",
            "frequency": 1
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
        },
        }
    
    
def train_routine(model: MLPClassifierWrapper, train_dataloader: torchdata.DataLoader, val_dataloader: torchdata.DataLoader, test_dataloader: torchdata.DataLoader, save_dir: Path):
    '''train routine for any train loader/val loader'''
    valcp = ModelCheckpoint(filename='{epoch}-{val_loss:.2f}', monitor='val_loss', mode='min')
    early_stop = EarlyStopping(monitor="val_loss", mode="min",check_finite=True, patience=10,  )
    trainer = pt.Trainer(logger = CSVLogger(str(save_dir)), accelerator='gpu', default_root_dir=str(save_dir), callbacks = [valcp, early_stop], max_epochs = 200)
    trainer.test(model, dataloaders = test_dataloader, verbose = True)
    trainer.fit(model = model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    return trainer

def test_routine(model: MLPClassifierWrapper, trainer: pt.Trainer, test_dataloader: torchdata.DataLoader, ckpt_path: Path):
    
    if ckpt_path == None:
        ckpt_path = 'best'
    trainer.test(dataloaders = test_dataloader, ckpt_path=str(ckpt_path), verbose = True)
    
    

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description = 'Trains a MLP classifier')
    parser.add_argument('-root', default = '..', help = 'Root path')
    parser.add_argument('-m', dest = 'model', help = 'Type of model: vggface2 or casia or vgg16')
    
    args = parser.parse_args()
    
    # seed for reproducibility
    pt.seed_everything(42)
        
    root = Path(args.root)
    
    # check trained model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Using {device} for inference')

    if args.model == 'vggface2' or args.model == 'casia':    
        input_features = 512
    elif args.model == 'vgg16':
        input_features = 4096
    else:
        raise ValueError("Model is not vggface2 or casia")
    
    # celeb results
    
    results_dir = root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    celeb_dir = results_dir / f'celeb_results_{args.model}'
    celeb_dir.mkdir(exist_ok=True)
    
    # load celeb data
    celeb_data = EmbeddingData(data_dir_name=f'celeb_embeddings_{args.model}', root = str(root), device= device, sample = False)
    
    # split data
    train_len = int(.8 * len(celeb_data))
    val_len = int(.10 * len(celeb_data))
    test_len = len(celeb_data) - train_len - val_len
    train_data, val_data, test_data = torchdata.random_split(celeb_data, lengths = [train_len, val_len, test_len])
    
    # loaders
    train_loader = torchdata.DataLoader(train_data, batch_size = 128, shuffle = True)
    val_loader = torchdata.DataLoader(val_data, batch_size = 256, shuffle = False)
    test_loader = torchdata.DataLoader(test_data, batch_size = 256, shuffle = False)
    
    # model 
    model = MLPClassifierWrapper(input_features = input_features)
    
    # train and test
    trainer = train_routine(model, train_loader, val_loader, test_loader, save_dir = celeb_dir)
    test_routine(model, trainer, test_loader, ckpt_path = 'best')
    