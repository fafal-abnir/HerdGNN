import torch
import time
import torch.nn.functional as F
from scipy.stats import norm
from dataclasses import dataclass
from torchmetrics.classification import BinaryAveragePrecision, BinaryAUROC
from torch.nn import BCEWithLogitsLoss

import pytorch_lightning as L


def merge_gaussians(mu1, sigma1, mu2, sigma2, weight):
    merged_mean = (1 - weight) * mu1 + weight * mu2
    var1 = sigma1 ** 2
    var2 = sigma2 ** 2
    merged_var = ((1 - weight) * (var1 + (mu1 - merged_mean) ** 2)
                  + weight * (var2 + (mu2 - merged_mean) ** 2))

    merged_std = merged_var.sqrt() if isinstance(merged_var, torch.Tensor) else merged_var ** 0.5

    return merged_mean, merged_std


class LightningNodeGNN(L.LightningModule):
    def __init__(self, model, learning_rate, alpha: float = 0.1, anomaly_loss_margin: float = 4.0,
                 blend_factor: float = 0.9):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.metric_avgpr = BinaryAveragePrecision()
        self.metric_auroc = BinaryAUROC()
        self.hparams.alpha = alpha
        self.anomaly_loss_margin = anomaly_loss_margin
        self.blend_factor = blend_factor
        self.normal_as_mean = 0
        self.normal_as_std = 1
        self.loss_fn = BCEWithLogitsLoss()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.train_start_time = None
        self.epoch_times = []
        self.epoch_start_time = None

    def reset_loss(self, loss):
        self.loss_fn = loss()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        pred, anomaly_scores, current_embeddings, h = self.model(x, edge_index)
        return pred, anomaly_scores, current_embeddings, h

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        return optimizer

    def _shared_step(self, batch):
        start_time = time.time()
        mask = batch.node_mask
        labels = batch.y[batch.node_mask]
        num_pos = (labels == 1).sum().float()
        num_neg = (labels == 0).sum().float()
        pos_weight = num_neg / (num_pos + 1e-6)

        pred, anomaly_scores, _, _ = self.forward(batch)
        # classification loss
        loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight)
        bce_loss = loss_fn(pred[mask], labels.type_as(pred))
        # anomaly loss
        masked_anomaly_scores = anomaly_scores[mask]
        anomaly_loss = self.anomaly_loss(masked_anomaly_scores, labels, self.anomaly_loss_margin)
        self.log("bce_loss", bce_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log("anomaly_loss", anomaly_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        total_loss = bce_loss + self.hparams.alpha * anomaly_loss
        self.log("total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        pred_cont = torch.sigmoid(pred)
        auc_roc = self.metric_auroc(pred_cont[batch.node_mask], batch.y[batch.node_mask].int())
        avg_pr = self.metric_avgpr(pred_cont[batch.node_mask], batch.y[batch.node_mask].int())
        num_total = labels.numel()
        num_pos = (labels == 1).sum().item()
        skew_ratio = num_pos / num_total
        aucpr_min = 1 + ((1 - skew_ratio) * torch.log(torch.tensor(1 - skew_ratio))) / skew_ratio
        aucnpr = (avg_pr - aucpr_min) / (1 - aucpr_min)
        if self.current_epoch + 1 == self.trainer.max_epochs:
            normal_scores = masked_anomaly_scores[labels == 0]
            mu, sigma = merge_gaussians(self.normal_as_mean, self.normal_as_std, normal_scores.mean().item(),
                                        normal_scores.std(unbiased=False).item(), self.blend_factor)
            self.normal_as_mean = mu
            self.normal_as_std = sigma
            self.anomaly_loss_margin = norm.ppf(1 - skew_ratio / 2, loc=mu, scale=sigma)
            print(self.anomaly_loss_margin)
        elapsed_time = time.time() - start_time
        self.log("time_sec", elapsed_time, on_step=False, on_epoch=True, prog_bar=True)
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e6  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1e6  # MB
            self.log("gpu_memory_allocated_MB", memory_allocated, prog_bar=True, on_step=False, on_epoch=True)
            self.log("gpu_memory_reserved_MB", memory_reserved, prog_bar=True, on_step=False, on_epoch=True)

        return total_loss, avg_pr, aucnpr, auc_roc

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

    def on_train_start(self):
        self.train_start_time = time.time()

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)

        self.log(
            "epoch_time_sec",
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

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
        _, _, _, node_embeddings = self.forward(batch)
        return node_embeddings


class LightningEdgeGNN(L.LightningModule):
    def __init__(self, model, learning_rate, alpha: float = 0.1, anomaly_loss_margin: float = 4.0,
                 blend_factor: float = 0.9):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.metric_avgpr = BinaryAveragePrecision()
        self.metric_auroc = BinaryAUROC()
        self.hparams.alpha = alpha
        self.anomaly_loss_margin = anomaly_loss_margin
        self.blend_factor = blend_factor
        self.normal_as_mean = 0
        self.normal_as_std = 1
        self.loss_fn = BCEWithLogitsLoss()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.epoch_start_time = None

    def reset_loss(self, loss):
        self.loss_fn = loss()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_label_index = data.edge_label_index
        edge_attr = data.edge_attr
        pred, anomaly_scores, current_embeddings = self.model(x, edge_index, edge_label_index, edge_attr)
        return pred, anomaly_scores, current_embeddings

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        return optimizer

    def on_train_start(self):
        print(f"================================= Model is on: {self.device}")

    def _shared_step(self, batch):
        start_time = time.time()
        labels = batch.y
        num_pos = (labels == 1).sum().float()
        num_neg = (labels == 0).sum().float()
        pos_weight = num_neg / (num_pos + 1e-6)
        pred, anomaly_scores, _ = self.forward(batch)
        # classification loss
        loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight)
        bce_loss = loss_fn(pred, labels.type_as(pred))
        # anomaly loss
        masked_anomaly_scores = anomaly_scores
        anomaly_loss = self.anomaly_loss(masked_anomaly_scores, labels, self.anomaly_loss_margin)
        # total loss
        self.log("bce_loss", bce_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log("anomaly_loss", anomaly_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        total_loss = bce_loss + self.hparams.alpha * anomaly_loss
        self.log("total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        pred_cont = torch.sigmoid(pred)
        avg_pr = self.metric_avgpr(pred_cont, batch.y.int())
        auc_roc = self.metric_auroc(pred_cont, batch.y.int())
        num_total = labels.numel()
        num_pos = (labels == 1).sum().item()
        skew_ratio = num_pos / float(num_total)
        aucpr_min = 1 + ((1 - skew_ratio) * torch.log(torch.tensor(1 - skew_ratio))) / skew_ratio
        aucnpr = (avg_pr - aucpr_min) / (1 - aucpr_min)
        if self.current_epoch + 1 == self.trainer.max_epochs:
            normal_scores = masked_anomaly_scores[labels == 0]
            mu, sigma = merge_gaussians(self.normal_as_mean, self.normal_as_std, normal_scores.mean().item(),
                                        normal_scores.std(unbiased=False).item(), self.blend_factor)
            self.normal_as_mean = mu
            self.normal_as_std = sigma
            self.anomaly_loss_margin = norm.ppf(1 - skew_ratio / 2, loc=mu, scale=sigma)
            print(f"Anomaly loss margin:{self.anomaly_loss_margin}")
        elapsed_time = time.time() - start_time
        self.log("time_sec", elapsed_time, on_step=False, on_epoch=True, prog_bar=True)
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e6  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1e6  # MB
            self.log("gpu_memory_allocated_MB", memory_allocated, prog_bar=True, on_step=False, on_epoch=True)
            self.log("gpu_memory_reserved_MB", memory_reserved, prog_bar=True, on_step=False, on_epoch=True)

        return total_loss, avg_pr, aucnpr, auc_roc

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
