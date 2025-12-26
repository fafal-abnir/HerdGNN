import time
import torch
from torchmetrics.classification import BinaryAveragePrecision, BinaryAUROC
from torch.nn import BCEWithLogitsLoss

import pytorch_lightning as L


class LightningNodeGNN(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.metric_avgpr = BinaryAveragePrecision()
        self.metric_auroc = BinaryAUROC()
        self.loss_fn = BCEWithLogitsLoss()
        self.save_hyperparameters(ignore=["model"])
        self.automatic_optimization = False
        self.epoch_start_time = None

    def reset_loss(self, loss):
        self.loss_fn = loss()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_ages = data.edge_ages
        pred, embeddings = self.model(x, edge_index, edge_ages)
        return pred, embeddings

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        return optimizer

    def _shared_step(self, batch):
        start_time = time.time()
        mask = batch.node_mask
        labels = batch.y[mask]
        num_pos = (labels == 1).sum().float()
        num_neg = (labels == 0).sum().float()
        pos_weight = num_neg / (num_pos + 1e-6)

        pred, _ = self.forward(batch)
        # classification loss
        loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight)
        bce_loss = loss_fn(pred[mask], labels.type_as(pred))
        # anomaly loss
        # total loss
        self.log("bce_loss", bce_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        # total_loss = bce_loss + self.hparams.alpha * anomaly_loss
        # self.log("total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        pred_cont = torch.sigmoid(pred)
        avg_pr = self.metric_avgpr(pred_cont[batch.node_mask], batch.y[batch.node_mask].int())
        auc_roc = self.metric_auroc(pred_cont[batch.node_mask], batch.y[batch.node_mask].int())
        num_total = labels.numel()
        num_pos = (labels == 1).sum().item()
        skew_ratio = num_pos / float(num_total)
        aucpr_min = 1 + ((1 - skew_ratio) * torch.log(torch.tensor(1 - skew_ratio))) / skew_ratio
        aucnpr = (avg_pr - aucpr_min) / (1 - aucpr_min)
        elapsed_time = time.time() - start_time
        self.log("time_sec", elapsed_time, on_step=False, on_epoch=True, prog_bar=True)
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e6  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1e6  # MB
            self.log("gpu_memory_allocated_MB", memory_allocated, prog_bar=True, on_step=False, on_epoch=True)
            self.log("gpu_memory_reserved_MB", memory_reserved, prog_bar=True, on_step=False, on_epoch=True)

        return bce_loss, avg_pr, aucnpr, auc_roc

    def training_step(self, batch, batch_idx):
        loss, avg_pr, aucnpr, auc_roc = self._shared_step(batch)
        start_time = time.time()
        optimizer = self.optimizers()  # Get the optimizer
        self.manual_backward(loss, retain_graph=True)  # Manually handle backward pass
        optimizer.step()  # Update the model parameters
        optimizer.zero_grad()  # Zero the gradients for the next step
        backprop_time = time.time() - start_time
        self.log("backprop_time", backprop_time, on_step=False, on_epoch=True, prog_bar=True, logger=True)
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
        start_time = time.time()
        loss, avg_pr, aucnpr, auc_roc = self._shared_step(batch)
        batch_size = batch[0].size(0)
        forward_time = time.time() - start_time
        throughput = batch_size / forward_time
        self.log("test_avg_pr", avg_pr)
        self.log("test_aucnpr", aucnpr)
        self.log("test_au_roc", auc_roc)
        self.log("test_loss", loss)
        self.log("test_forward_time", forward_time)
        self.log("test_samples_count", batch_size)
        self.log("test_throughput_samples_per_sec", throughput)

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time

        self.log(
            "epoch_time_sec",
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

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
        self.save_hyperparameters(ignore=["model"])
        self.automatic_optimization = False
        self.epoch_start_time = None

    def reset_loss(self, loss):
        self.loss_fn = loss()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_label_index = data.edge_label_index
        edge_attr = data.edge_attr
        edge_ages = data.edge_ages
        pred, embeddings = self.model(x, edge_index, edge_label_index, edge_attr, edge_ages)
        return pred, embeddings

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        return optimizer

    def _shared_step(self, batch):
        start_time = time.time()
        # mask = batch.node_mask
        # labels = batch.y[mask]
        labels = batch.y
        num_pos = (labels == 1).sum().float()
        num_neg = (labels == 0).sum().float()
        pos_weight = num_neg / (num_pos + 1e-6)
        pred, _ = self.forward(batch)
        # classification loss
        loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight)
        bce_loss = loss_fn(pred, labels.type_as(pred))
        self.log("bce_loss", bce_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        pred_cont = torch.sigmoid(pred)
        avg_pr = self.metric_avgpr(pred_cont, batch.y.int())
        auc_roc = self.metric_auroc(pred_cont, batch.y.int())
        num_total = labels.numel()
        num_pos = (labels == 1).sum().item()
        skew_ratio = num_pos / float(num_total)
        aucpr_min = 1 + ((1 - skew_ratio) * torch.log(torch.tensor(1 - skew_ratio))) / skew_ratio
        aucnpr = (avg_pr - aucpr_min) / (1 - aucpr_min)
        elapsed_time = time.time() - start_time
        self.log("time_sec", elapsed_time, on_step=False, on_epoch=True, prog_bar=True)
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e6  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1e6  # MB
            self.log("gpu_memory_allocated_MB", memory_allocated, prog_bar=True, on_step=False, on_epoch=True)
            self.log("gpu_memory_reserved_MB", memory_reserved, prog_bar=True, on_step=False, on_epoch=True)

        return bce_loss, avg_pr, aucnpr, auc_roc

    def training_step(self, batch, batch_idx):
        loss, avg_pr, aucnpr, auc_roc = self._shared_step(batch)
        start_time = time.time()
        optimizer = self.optimizers()  # Get the optimizer
        self.manual_backward(loss, retain_graph=True)  # Manually handle backward pass
        optimizer.step()  # Update the model parameters
        optimizer.zero_grad()  # Zero the gradients for the next step
        backprop_time = time.time() - start_time
        self.log("backprop_time", backprop_time, on_step=False, on_epoch=True, prog_bar=True, logger=True)
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
        start_time = time.time()
        loss, avg_pr, aucnpr, auc_roc = self._shared_step(batch)
        batch_size = batch[0].size(0) # for one batch
        forward_time = time.time() - start_time
        throughput = batch_size / forward_time
        self.log("test_avg_pr", avg_pr)
        self.log("test_aucnpr", aucnpr)
        self.log("test_au_roc", auc_roc)
        self.log("test_loss", loss)
        self.log("test_forward_time", forward_time)
        self.log("test_samples_count", batch_size)
        self.log("test_throughput_samples_per_sec", throughput)

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time

        self.log(
            "epoch_time_sec",
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def get_node_embeddings(self, batch):
        """Extracts node embeddings before and after training."""
        _, node_embeddings = self.forward(batch)
        return node_embeddings
