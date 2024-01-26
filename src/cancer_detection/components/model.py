import torch
import torch.nn as nn
import torchvision
from src.cancer_detection.entity.config_dataclasses import TrainingConfig
import pytorch_lightning as pl

# ---------------------------- transfer learning model----------------------- #
class vgg16_modified(nn.Module):
    def __init__(self, config: TrainingConfig):
        """
        Modified VGG16 model for transfer learning.

        Parameters:
        - config (TrainingConfig): Training configuration data class.
        """
        super().__init__()
        self.config = config
        self.model =  torchvision.models.vgg16(pretrained=True)
        self.model.classifier[6] = nn.Linear(
            in_features=self.model.classifier[6].in_features,
            out_features=self.config.params_num_classes
        )

        # Specify the layers for updating
        params_to_update = []
        update_params_name = ['classifier.6.weight', 'classifier.6.bias']
        for name, param in self.model.named_parameters():
            if name in update_params_name:
                param.requires_grad = True
                params_to_update.append(param)
            else:
                param.requires_grad = False
    
    def forward(self, batch):
        """Forward pass of the model."""
        return self.model(batch)
    

# ---------------------------- Lightning Module ----------------------------- #
class cancerClassifier(pl.LightningModule):
    def __init__(self, model: nn.Module, config: TrainingConfig) -> None:
        """
        Lightning Module for the cancer classifier.

        Parameters:
        - model (nn.Module): PyTorch model for cancer classification.
        - config (TrainingConfig): Training configuration data class.
        """
        super().__init__()
        self.config = config
        self.model = model
        # define loss function
        self.criterion = nn.CrossEntropyLoss()
        

    def forward(self, batch: torch.tensor) -> torch.tensor:
        """Forward pass of the model to return output predictions."""
        return self.model(batch)


    def training_step(self, batch: torch.tensor, batch_idx: int) -> torch.tensor:
        """
        Perform a single training step, returning the loss on a training batch.

        Parameters:
        - batch (torch.tensor): Training batch.
        - batch_idx (int): Index of the current batch.

        Returns:
        torch.tensor: Loss on the training batch.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True, logger=True)

        preds = self.predict_step(logits)
        acc = torch.sum(preds == y).float()/len(y)
        self.log("train_acc", acc, prog_bar=True, logger=True)
        return loss
    

    def validation_step(self, batch: torch.tensor, batch_idx: int) -> None:
        """
        Perform a single validation step.

        Parameters:
        - batch (torch.tensor): Validation batch.
        - batch_idx (int): Index of the current batch.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log("valid_loss", loss, prog_bar=True, logger=True)

        preds = self.predict_step(logits)
        acc = torch.sum(preds == y).float()/len(y)
        self.log("val_acc", acc, prog_bar=True, logger=True)


    def test_step(self, batch: torch.tensor, batch_idx: int) -> None:
        """
        Perform a single test step.

        Parameters:
        - batch (torch.tensor): Test batch.
        - batch_idx (int): Index of the current batch.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log("test_loss", loss, prog_bar=True, logger=True)

        preds = self.predict_step(logits)
        acc = torch.sum(preds == y).float()/len(y)
        self.log("test_acc", acc, prog_bar=True, logger=True)


    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer to use for training.

        Returns:
        torch.optim.Optimizer: Optimizer for training.
        """
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config.params_learning_rate, # add learning rate here
        )


    def predict_step(self, logits: torch.tensor) -> torch.tensor: 
        """
        Perform the prediction step.

        Parameters:
        - logits (torch.tensor): Logits from the model.

        Returns:
        torch.tensor: Predicted labels.
        """
        preds = torch.argmax(logits, dim=1)
        return preds
