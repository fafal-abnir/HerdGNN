import time
import torch
from torchmetrics.classification import BinaryAveragePrecision, BinaryAUROC
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss

import pytorch_lightning as L


class LightningNodeGNN(L.LightningModule):
    def __init__(self, model, learning_rate, alpha: float = 0.1):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.metric_avgpr = BinaryAveragePrecision()
        self.metric_auroc = BinaryAUROC()
        self.hparams.alpha = alpha
        self.normal_as_mean = 0
        self.normal_as_std = 1
        self.loss_fn = BCEWithLogitsLoss()
        # self.loss_fn = BinaryAUROC()
        self.save_hyperparameters(ignore=["model"])
        self.automatic_optimization = False

    def reset_loss(self, loss):
        self.loss_fn = loss()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        pred, anomaly_scores, embeddings = self.model(x, edge_index)
        return pred, anomaly_scores, embeddings

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        return optimizer

    def _shared_step(self, batch):
        start_time = time.time()
        mask = batch.node_mask
        labels = batch.y[mask]
        pred, anomaly_scores, _ = self.forward(batch)
        # classification loss
        bce_loss = self.loss_fn(pred[mask], labels.type_as(pred))
        # anomaly loss
        masked_anomaly_scores = anomaly_scores[mask]
        anomaly_loss = self.anomaly_loss(masked_anomaly_scores, labels)
        # total loss
        self.log("bce_loss", bce_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log("anomaly_loss", anomaly_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        total_loss = bce_loss + self.hparams.alpha * anomaly_loss
        self.log("total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        if self.current_epoch + 1 == self.trainer.max_epochs:
            normal_scores = masked_anomaly_scores[labels == 0]
            self.normal_as_mean = normal_scores.mean().item()
            self.normal_as_std = normal_scores.std(unbiased=False).item()
        pred_cont = torch.sigmoid(pred)
        avg_pr = self.metric_avgpr(pred_cont[batch.node_mask], batch.y[batch.node_mask].int())
        auc_roc = self.metric_auroc(pred_cont[batch.node_mask], batch.y[batch.node_mask].int())
        elapsed_time = time.time() - start_time
        self.log("time_sec", elapsed_time, on_step=False, on_epoch=True, prog_bar=True)
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e6  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1e6  # MB
            self.log("gpu_memory_allocated_MB", memory_allocated, prog_bar=True, on_step=False, on_epoch=True)
            self.log("gpu_memory_reserved_MB", memory_reserved, prog_bar=True, on_step=False, on_epoch=True)

        return total_loss, avg_pr, auc_roc

    def training_step(self, batch, batch_idx):
        loss, avg_pr, auc_roc = self._shared_step(batch)
        optimizer = self.optimizers()  # Get the optimizer
        self.manual_backward(loss, retain_graph=True)  # Manually handle backward pass
        optimizer.step()  # Update the model parameters
        optimizer.zero_grad()  # Zero the gradients for the next step
        self.log("train_avg_pr", avg_pr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_au_roc", auc_roc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, avg_pr, auc_roc = self._shared_step(batch)
        self.log("val_avg_pr", avg_pr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_au_roc", auc_roc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss, avg_pr, auc_roc = self._shared_step(batch)
        self.log("test_avg_pr", avg_pr)
        self.log("test_au_roc", auc_roc)
        self.log("test_loss", loss)

    def compute_deviation_score(self, scores: torch.Tensor) -> torch.Tensor:
        mu = self.normal_as_mean
        sigma = self.normal_as_std + 1e-6
        return torch.abs(scores - mu) / sigma

    def anomaly_loss(self, scores: torch.Tensor, labels: torch.Tensor, margin: float = 4.0):
        dev = self.compute_deviation_score(scores)
        normal_loss = dev[labels == 0]
        anomaly_loss = F.relu(margin - dev[labels == 1])
        loss = 0.0
        if len(normal_loss) > 0:
            loss += normal_loss.mean()
        if len(anomaly_loss) > 0:
            loss += anomaly_loss.mean()
        return loss

    def get_node_embeddings(self, batch):
        """Extracts node embeddings before and after training."""
        _, node_embeddings = self.forward(batch)
        return node_embeddings

class LightningEdgeGNN(L.LightningModule):
    def __init__(self, model, learning_rate, alpha: float = 0.1):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.metric_avgpr = BinaryAveragePrecision()
        self.metric_auroc = BinaryAUROC()
        self.hparams.alpha = alpha
        self.normal_as_mean = 0
        self.normal_as_std = 1
        self.loss_fn = BCEWithLogitsLoss()
        self.save_hyperparameters(ignore=["model"])
        self.automatic_optimization = False

    def reset_loss(self, loss):
        self.loss_fn = loss()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_label_index = data.edge_label_index
        edge_attr = data.edge_attr
        pred, anomaly_scores, embeddings = self.model(x, edge_index,edge_label_index, edge_attr)
        return pred, anomaly_scores, embeddings

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        return optimizer

    def _shared_step(self, batch):
        start_time = time.time()
        # mask = batch.node_mask
        # labels = batch.y[mask]
        labels = batch.y
        pred, anomaly_scores, _ = self.forward(batch)
        # classification loss
        bce_loss = self.loss_fn(pred, labels.type_as(pred))
        # anomaly loss
        masked_anomaly_scores = anomaly_scores
        anomaly_loss = self.anomaly_loss(masked_anomaly_scores, labels)
        # total loss
        self.log("bce_loss", bce_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log("anomaly_loss", anomaly_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        total_loss = bce_loss + self.hparams.alpha * anomaly_loss
        self.log("total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        if self.current_epoch + 1 == self.trainer.max_epochs:
            normal_scores = masked_anomaly_scores[labels == 0]
            self.normal_as_mean = normal_scores.mean().item()
            self.normal_as_std = normal_scores.std(unbiased=False).item()
        pred_cont = torch.sigmoid(pred)
        avg_pr = self.metric_avgpr(pred_cont, batch.y.int())
        auc_roc = self.metric_auroc(pred_cont, batch.y.int())
        elapsed_time = time.time() - start_time
        self.log("time_sec", elapsed_time, on_step=False, on_epoch=True, prog_bar=True)
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e6  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1e6  # MB
            self.log("gpu_memory_allocated_MB", memory_allocated, prog_bar=True, on_step=False, on_epoch=True)
            self.log("gpu_memory_reserved_MB", memory_reserved, prog_bar=True, on_step=False, on_epoch=True)

        return total_loss, avg_pr, auc_roc

    def training_step(self, batch, batch_idx):
        loss, avg_pr, auc_roc = self._shared_step(batch)
        optimizer = self.optimizers()  # Get the optimizer
        self.manual_backward(loss, retain_graph=True)  # Manually handle backward pass
        optimizer.step()  # Update the model parameters
        optimizer.zero_grad()  # Zero the gradients for the next step
        self.log("train_avg_pr", avg_pr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_au_roc", auc_roc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, avg_pr, auc_roc = self._shared_step(batch)
        self.log("val_avg_pr", avg_pr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_au_roc", auc_roc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss, avg_pr, auc_roc = self._shared_step(batch)
        self.log("test_avg_pr", avg_pr)
        self.log("test_au_roc", auc_roc)
        self.log("test_loss", loss)

    def compute_deviation_score(self, scores: torch.Tensor) -> torch.Tensor:
        mu = self.normal_as_mean
        sigma = self.normal_as_std + 1e-6
        return torch.abs(scores - mu) / sigma

    def anomaly_loss(self, scores: torch.Tensor, labels: torch.Tensor, margin: float = 4.0):
        dev = self.compute_deviation_score(scores)
        normal_loss = dev[labels == 0]
        anomaly_loss = F.relu(margin - dev[labels == 1])
        loss = 0.0
        if len(normal_loss) > 0:
            loss += normal_loss.mean()
        if len(anomaly_loss) > 0:
            loss += anomaly_loss.mean()
        return loss

    def get_node_embeddings(self, batch):
        """Extracts node embeddings before and after training."""
        _, node_embeddings = self.forward(batch)
        return node_embeddings
