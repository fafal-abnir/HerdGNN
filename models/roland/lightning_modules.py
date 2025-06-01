import torch
import time
from dataclasses import dataclass
from torchmetrics.classification import BinaryAveragePrecision, BinaryAUROC
from torch.nn import BCEWithLogitsLoss

import pytorch_lightning as L


@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    metric: str = "torchmetrics.classification.BinaryAveragePrecision"
    loss_fn: str = "torch.nn.BCEWithLogitsLoss"


class LightningNodeGNN(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.metric_avgpr = BinaryAveragePrecision()
        self.metric_auroc = BinaryAUROC()

        self.loss_fn = BCEWithLogitsLoss()
        self.save_hyperparameters()
        self.automatic_optimization = False

    def reset_loss(self, loss):
        self.loss_fn = loss()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        pred, current_embeddings = self.model(x, edge_index)
        return pred, current_embeddings

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        return optimizer

    def _shared_step(self, batch):
        start_time = time.time()
        pred, _ = self.forward(batch)
        loss = self.loss_fn(pred[batch.node_mask], batch.y[batch.node_mask].type_as(pred))
        pred_cont = torch.sigmoid(pred)
        labels = batch.y[batch.node_mask]
        auc_roc = self.metric_auroc(pred_cont[batch.node_mask], batch.y[batch.node_mask].int())
        avg_pr = self.metric_avgpr(pred_cont[batch.node_mask], batch.y[batch.node_mask].int())
        num_total = labels.numel()
        num_pos = (labels ==1).sum().item()
        skew_ratio = num_pos/num_total
        aucpr_min = 1 + ((1 - skew_ratio) * torch.log(torch.tensor(1 - skew_ratio))) / skew_ratio
        aucnpr = (avg_pr-aucpr_min)/(1-aucpr_min)
        elapsed_time = time.time() - start_time
        self.log("time_sec", elapsed_time, on_step=False, on_epoch=True, prog_bar=True)
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e6  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1e6  # MB
            self.log("gpu_memory_allocated_MB", memory_allocated, prog_bar=True, on_step=False, on_epoch=True)
            self.log("gpu_memory_reserved_MB", memory_reserved, prog_bar=True, on_step=False, on_epoch=True)

        return loss, avg_pr, aucnpr, auc_roc

    def training_step(self, batch, batch_idx):
        loss, avg_pr, aucnpr, auc_roc = self._shared_step(batch)
        optimizer = self.optimizers()  # Get the optimizer
        self.manual_backward(loss, retain_graph=True)  # Manually handle backward pass
        optimizer.step()  # Update the model parameters
        optimizer.zero_grad()  # Zero the gradients for the next step
        self.log("train_avg_pr", avg_pr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_aucnpr", aucnpr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_au_roc", auc_roc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, avg_pr, aucnpr, auc_roc = self._shared_step(batch)
        self.log("val_avg_pr", avg_pr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_aucnpr", aucnpr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_au_roc", auc_roc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss, avg_pr, aucnpr, auc_roc= self._shared_step(batch)
        self.log("test_avg_pr", avg_pr)
        self.log("test_aucnpr", aucnpr)
        self.log("test_au_roc", auc_roc)
        self.log("test_loss", loss)

    def get_node_embeddings(self, batch):
        """Extracts node embeddings before and after training."""
        _, node_embeddings = self.forward(batch)
        return node_embeddings


class LightningEdgeGNN(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.metric_avgpr = BinaryAveragePrecision()
        self.metric_auroc = BinaryAUROC()

        self.loss_fn = BCEWithLogitsLoss()
        self.save_hyperparameters()
        self.automatic_optimization = False

    def reset_loss(self, loss):
        self.loss_fn = loss()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_label_index = data.edge_label_index
        edge_attr = data.edge_attr
        pred, current_embeddings = self.model(x, edge_index, edge_label_index, edge_attr)
        return pred, current_embeddings

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        return optimizer

    def on_train_start(self):
        print(f"================================= Model is on: {self.device}")

    def _shared_step(self, batch):
        start_time = time.time()
        pred, _ = self.forward(batch)
        loss = self.loss_fn(pred, batch.y.type_as(pred))
        labels =  batch.y
        pred_cont = torch.sigmoid(pred)
        avg_pr = self.metric_avgpr(pred_cont, batch.y.int())
        auc_roc = self.metric_auroc(pred_cont, batch.y.int())
        num_total = labels.numel()
        num_pos = (labels == 1).sum().item()
        skew_ratio = num_pos / float(num_total)
        aucpr_min = 1 + ((1 - skew_ratio) * torch.log(torch.tensor(1 - skew_ratio))) / skew_ratio
        aucnpr = (avg_pr - aucpr_min) / (1 - aucpr_min)
        elapsed_time = time.time() - start_time
        self.log("forward_time_sec", elapsed_time, on_step=False, on_epoch=True, prog_bar=True)
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e6  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1e6  # MB
            self.log("gpu_memory_allocated_MB", memory_allocated, prog_bar=True, on_step=False, on_epoch=True)
            self.log("gpu_memory_reserved_MB", memory_reserved, prog_bar=True, on_step=False, on_epoch=True)

        return loss, avg_pr, aucnpr, auc_roc

    def training_step(self, batch, batch_idx):
        loss, avg_pr, aucnpr, auc_roc = self._shared_step(batch)
        start_time = time.time()
        optimizer = self.optimizers()  # Get the optimizer
        self.manual_backward(loss, retain_graph=True)  # Manually handle backward pass
        optimizer.step()  # Update the model parameters
        optimizer.zero_grad()  # Zero the gradients for the next step
        elapsed_time = time.time() - start_time
        self.log("backward_time_sec", elapsed_time, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_avg_pr", avg_pr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_aucnpr", aucnpr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_au_roc", auc_roc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, avg_pr, aucnpr, auc_roc = self._shared_step(batch)
        self.log("val_avg_pr", avg_pr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_aucnpr", aucnpr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_au_roc", auc_roc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss, avg_pr, aucnpr, auc_roc = self._shared_step(batch)
        self.log("test_avg_pr", avg_pr)
        self.log("test_aucnpr", aucnpr)
        self.log("test_au_roc", auc_roc)
        self.log("test_loss", loss)

    def get_node_embeddings(self, batch):
        """Extracts node embeddings before and after training."""
        _, node_embeddings = self.forward(batch)
        return node_embeddings
