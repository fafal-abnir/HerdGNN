import pytorch_lightning as L
import torch.optim
import time

from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision


class LightningNodeGNN(L.LightningModule):
    def __init__(self, model, learning_rate, lam):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.metric_avgpr = BinaryAveragePrecision()
        self.metric_auroc = BinaryAUROC()
        self.save_hyperparameters(ignore=["model"])
        self.automatic_optimization = False

        self.lam = lam

    def reset_loss(self, loss):
        self.loss_fn = loss

    @staticmethod
    def auc_loss(normal_embeddings: torch, anomaly_embeddings):
        diffs = normal_embeddings.unsqueeze(1) - anomaly_embeddings.unsqueeze(0)
        distances = torch.norm(diffs, dim=2)
        mean_distance = distances.mean()
        return mean_distance

    @staticmethod
    def nor_loss(normal_embeddings):
        norms = torch.norm(normal_embeddings, dim=1)
        return norms.mean()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        pred, embeddings = self.model(x, edge_index)
        return pred, embeddings

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)
        return optimizer

    def _shared_step(self, batch):
        start_time = time.time()
        mask = batch.node_mask
        labels = batch.y[mask]
        pred, embeddings = self.forward(batch)
        current_embeddings = embeddings[batch.node_mask]
        zero_embeddings = current_embeddings[labels == 0]
        one_embeddings = current_embeddings[labels == 1]
        # nor loss
        nor_loss = self.nor_loss(zero_embeddings)
        self.log("nor_loss", nor_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        # auc loss
        auc_loss = self.auc_loss(zero_embeddings, one_embeddings)
        self.log("auc_loss", auc_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        # total loss
        total_loss = nor_loss - self.lam * auc_loss
        self.log("total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
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

        return total_loss, avg_pr, aucnpr, auc_roc

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
        loss, avg_pr, aucnpr, auc_roc = self._shared_step(batch)
        self.log("test_avg_pr", avg_pr)
        self.log("test_aucnpr", aucnpr)
        self.log("test_au_roc", auc_roc)
        self.log("test_loss", loss)

    def get_node_embeddings(self, batch):
        """Extracts node embeddings before and after training."""
        _, node_embeddings = self.forward(batch)
        return node_embeddings
