import time
import torch
import random
from scipy.stats import norm
from torchmetrics.classification import BinaryAveragePrecision, BinaryAUROC
from torch.nn.utils.stateless import functional_call
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss

import pytorch_lightning as L


class LightningNodeGNN(L.LightningModule):
    def __init__(self, model, learning_rate, beta: float = 0.89, training_window_size: int = 5, drop_snap: float = 0.3,
                 clip_grad_norm: float = 1.0):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.beta = beta
        self.train_window_size = training_window_size
        self.drop_snap = drop_snap
        self.clip_grad_norm = clip_grad_norm
        self.metric_avgpr = BinaryAveragePrecision()
        self.metric_auroc = BinaryAUROC()
        self.loss_fn = BCEWithLogitsLoss()
        self.save_hyperparameters(ignore=["model"])
        self.automatic_optimization = False
        self.param_names = [name for name, _ in self.model.named_parameters()]
        self.S_dw = [torch.zeros_like(p) for _, p in self.model.named_parameters()]
        self.epoch_start_time = None

    def reset_loss(self, loss):
        self.loss_fn = loss()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        pred, embeddings = self.model(x, edge_index)
        return pred, embeddings

    def forward_with_fast_weights(self, data, fast_weights):
        """
        Forward pass using a custom parameter set (fast_weights) via functional_call.
        Model forward: model(x, edge_index).
        """
        x = data.x
        edge_index = data.edge_index
        param_dict = {name: w for name, w in zip(self.param_names, fast_weights)}

        pred, embeddings = functional_call(
            self.model,
            param_dict,
            (x, edge_index),
        )
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
        # self.log("bce_loss", bce_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        pred_cont = torch.sigmoid(pred)
        avg_pr = self.metric_avgpr(pred_cont[batch.node_mask], batch.y[batch.node_mask].int())
        auc_roc = self.metric_auroc(pred_cont[batch.node_mask], batch.y[batch.node_mask].int())
        num_total = labels.numel()
        num_pos = (labels == 1).sum().item()
        skew_ratio = num_pos / float(num_total)
        aucpr_min = 1 + ((1 - skew_ratio) * torch.log(torch.tensor(1 - skew_ratio))) / skew_ratio
        aucnpr = (avg_pr - aucpr_min) / (1 - aucpr_min)
        elapsed_time = time.time() - start_time
        # self.log("time_sec", elapsed_time, on_step=False, on_epoch=True, prog_bar=True)
        return bce_loss, avg_pr, aucnpr, auc_roc

    def _step_with_fast_weights(self, batch, fast_weights):
        """
        inner loss step logic as _shared_step but:
        - uses forward_with_fast_weights
        - does NOT log anything
        Returns: loss, avg_pr, aucnpr, auc_roc
        """
        mask = batch.node_mask
        labels = batch.y[mask]
        num_pos = (labels == 1).sum().float()
        num_neg = (labels == 0).sum().float()
        pos_weight = num_neg / (num_pos + 1e-6)
        pred, _ = self.forward_with_fast_weights(batch, fast_weights)
        # elapsed_time = time.time() - start_time
        # self.log("time_sec", elapsed_time, on_step=False, on_epoch=True, prog_bar=True)
        loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight)
        bce_loss = loss_fn(pred[mask], labels.type_as(pred))

        pred_cont = torch.sigmoid(pred)
        avg_pr = self.metric_avgpr(pred_cont[batch.node_mask], batch.y[batch.node_mask].int())
        auc_roc = self.metric_auroc(pred_cont[batch.node_mask], batch.y[batch.node_mask].int())

        num_total = labels.numel()
        num_pos = (labels == 1).sum().item()
        skew_ratio = num_pos / float(num_total)
        aucpr_min = 1 + ((1 - skew_ratio) * torch.log(torch.tensor(1 - skew_ratio))) / skew_ratio
        aucnpr = (avg_pr - aucpr_min) / (1 - aucpr_min)

        return bce_loss, avg_pr, aucnpr, auc_roc

    def training_step(self, train_graph_list, batch_idx):
        start_time = time.time()
        optimizer = self.optimizers()
        device = self.device
        self.S_dw = [s.to(device) for s in self.S_dw]
        i = 0
        aucnpr_window_means = []
        avg_pr_window_means = []
        auc_roc_window_means = []
        last_window_loss = None
        while i < len(train_graph_list) - self.train_window_size:
            if i != 0:
                i = random.randint(i, i + self.train_window_size)
            if i >= (len(train_graph_list) - self.train_window_size):
                break
            ds_window = train_graph_list[i:i + self.train_window_size]
            i += 1
            fast_weights = [p for _, p in self.model.named_parameters()]
            window_loss_sum = 0
            window_avg_pr_list = []
            window_aucnpr_list = []
            window_auc_roc_list = []
            for idx, data in enumerate(ds_window[:-2]):
                data = data.to(device)
                target = ds_window[idx + 1].to(device)
                inner_loss, _, _, _ = self._step_with_fast_weights(data, fast_weights)
                grads = torch.autograd.grad(inner_loss, fast_weights, retain_graph=False, create_graph=False)
                new_S_dw = []
                new_fast_weights = []
                for g, s, w in zip(grads, self.S_dw, fast_weights):
                    s_new = self.beta * s + (1.0 - self.beta) * (g * g)
                    step = self.learning_rate / (torch.sqrt(s_new) + 1e-8) * g
                    new_S_dw.append(s_new)
                    new_fast_weights.append(w - step)

                # for (name, w), g, s in zip(self.model.named_parameters(), grads, self.S_dw):
                #     if g is None:
                #         # Parameter not used in this inner_loss; keep as is
                #         new_S_dw.append(s)
                #         new_fast_weights.append(w)
                #         continue
                #     print(f"{name:40s} param={w.device}   S_dw={s.device}")
                #     s_new = self.beta * s + (1.0 - self.beta) * (g * g)
                #     step = self.learning_rate / (torch.sqrt(s_new) + 1e-8) * g
                #     new_S_dw.append(s_new)
                #     new_fast_weights.append(w - step)



                self.S_dw = new_S_dw
                fast_weights = new_fast_weights

                # window-aware "future" loss, use next snapshot as target
                val_loss, avg_pr, aucnpr, auc_roc = self._step_with_fast_weights(target, fast_weights)

                if random.random() > self.drop_snap:
                    window_loss_sum = window_loss_sum + val_loss
                    window_avg_pr_list.append(avg_pr.detach())
                    window_aucnpr_list.append(aucnpr.detach())
                    window_auc_roc_list.append(auc_roc.detach())

            # outer update, use aggregated future losses for this window
            if len(window_aucnpr_list) > 0: #WinGNN implementation from their code, a bit weird
                window_loss = window_loss_sum / len(window_aucnpr_list)

                optimizer.zero_grad()
                self.manual_backward(window_loss)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                optimizer.step()

                last_window_loss = window_loss
                aucnpr_window_means.append(torch.stack(window_aucnpr_list).mean().item())
                avg_pr_window_means.append(torch.stack(window_avg_pr_list).mean().item())
                auc_roc_window_means.append(torch.stack(window_auc_roc_list).mean().item())
        elapsed_time = time.time() - start_time
        self.log("backprop_time", elapsed_time, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if len(aucnpr_window_means) > 0:
            mean_aucnpr = float(torch.tensor(aucnpr_window_means).mean().item())
            mean_avg_pr = float(torch.tensor(avg_pr_window_means).mean().item())
            mean_auc_roc = float(torch.tensor(auc_roc_window_means).mean().item())
            self.log("train_avg_pr", mean_avg_pr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_aucnpr", mean_aucnpr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_au_roc", mean_auc_roc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_loss", last_window_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return last_window_loss
        else:
            # no window processed (e.g., very short sequence), should we return dummy loss
            dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
            self.log("train_aucnpr", 0.0, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_avg_pr", 0.0, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_aucnpr", 0.0, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_au_roc", 0.0, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return dummy_loss

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
    def __init__(self, model, learning_rate, beta: float = 0.89, training_window_size: int = 5, drop_snap: float = 0.3,
                 clip_grad_norm: float = 1.0):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.beta = beta
        self.train_window_size = training_window_size
        self.drop_snap = drop_snap
        self.clip_grad_norm = clip_grad_norm
        self.metric_avgpr = BinaryAveragePrecision()
        self.metric_auroc = BinaryAUROC()
        self.loss_fn = BCEWithLogitsLoss()
        self.save_hyperparameters(ignore=["model"])
        self.automatic_optimization = False
        self.param_names = [name for name, _ in self.model.named_parameters()]
        self.S_dw = [torch.zeros_like(p) for _, p in self.model.named_parameters()]
        self.epoch_start_time = None

    def reset_loss(self, loss):
        self.loss_fn = loss()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_label_index = data.edge_label_index
        edge_attr = data.edge_attr
        pred, embeddings = self.model(x, edge_index, edge_label_index, edge_attr)
        return pred, embeddings

    def forward_with_fast_weights(self, data, fast_weights):
        """
        Forward pass using a custom parameter set (fast_weights) via functional_call.
        Model forward: model(x, edge_index).
        """
        x = data.x
        edge_index = data.edge_index
        edge_label_index = data.edge_label_index
        edge_attr = data.edge_attr

        param_dict = {name: w for name, w in zip(self.param_names, fast_weights)}

        pred, embeddings = functional_call(
            self.model,
            param_dict,
            (x, edge_index, edge_label_index, edge_attr),
        )
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
        # bce_loss = self.loss_fn(pred, labels.type_as(pred))
        # self.log("bce_loss", bce_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        pred_cont = torch.sigmoid(pred)
        avg_pr = self.metric_avgpr(pred_cont, batch.y.int())
        auc_roc = self.metric_auroc(pred_cont, batch.y.int())
        num_total = labels.numel()
        num_pos = (labels == 1).sum().item()
        skew_ratio = num_pos / float(num_total)
        aucpr_min = 1 + ((1 - skew_ratio) * torch.log(torch.tensor(1 - skew_ratio))) / skew_ratio
        aucnpr = (avg_pr - aucpr_min) / (1 - aucpr_min)
        elapsed_time = time.time() - start_time
        # self.log("time_sec", elapsed_time, on_step=False, on_epoch=True, prog_bar=True)
        # if torch.cuda.is_available():
        #     memory_allocated = torch.cuda.memory_allocated() / 1e6  # MB
        #     memory_reserved = torch.cuda.memory_reserved() / 1e6  # MB
        #     self.log("gpu_memory_allocated_MB", memory_allocated, prog_bar=True, on_step=False, on_epoch=True)
        #     self.log("gpu_memory_reserved_MB", memory_reserved, prog_bar=True, on_step=False, on_epoch=True)

        return bce_loss, avg_pr, aucnpr, auc_roc

    def _step_with_fast_weights(self, batch, fast_weights):
        start_time = time.time()
        # mask = batch.node_mask
        # labels = batch.y[mask]
        labels = batch.y
        num_pos = (labels == 1).sum().float()
        num_neg = (labels == 0).sum().float()
        pos_weight = num_neg / (num_pos + 1e-6)
        pred, _ = self.forward_with_fast_weights(batch, fast_weights)
        loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight)
        bce_loss = loss_fn(pred, labels.type_as(pred))
        pred_cont = torch.sigmoid(pred)
        avg_pr = self.metric_avgpr(pred_cont, batch.y.int())
        auc_roc = self.metric_auroc(pred_cont, batch.y.int())
        num_total = labels.numel()
        num_pos = (labels == 1).sum().item()
        skew_ratio = num_pos / float(num_total)
        aucpr_min = 1 + ((1 - skew_ratio) * torch.log(torch.tensor(1 - skew_ratio))) / skew_ratio
        aucnpr = (avg_pr - aucpr_min) / (1 - aucpr_min)

        return bce_loss, avg_pr, aucnpr, auc_roc

    def training_step(self, train_graph_list, batch_idx):
        start_time = time.time()
        optimizer = self.optimizers()
        device = self.device
        i = 0
        self.S_dw = [s.to(device) for s in self.S_dw]
        aucnpr_window_means = []
        avg_pr_window_means = []
        auc_roc_window_means = []
        last_window_loss = None
        while i < len(train_graph_list) - self.train_window_size:
            if i != 0:
                i = random.randint(i, i + self.train_window_size)
            if i >= (len(train_graph_list) - self.train_window_size):
                break
            ds_window = train_graph_list[i:i + self.train_window_size]
            i += 1
            fast_weights = [p for _, p in self.model.named_parameters()]
            window_loss_sum = 0
            window_avg_pr_list = []
            window_aucnpr_list = []
            window_auc_roc_list = []
            for idx, data in enumerate(ds_window[:-2]):
                data = data.to(device)
                target = ds_window[idx + 1].to(device)
                inner_loss, _, _, _ = self._step_with_fast_weights(data, fast_weights)
                grads = torch.autograd.grad(inner_loss, fast_weights, retain_graph=False, create_graph=False)
                new_S_dw = []
                new_fast_weights = []
                for g, s, w in zip(grads, self.S_dw, fast_weights):
                    s_new = self.beta * s + (1.0 - self.beta) * (g * g)
                    step = self.learning_rate / (torch.sqrt(s_new) + 1e-8) * g
                    new_S_dw.append(s_new)
                    new_fast_weights.append(w - step)

                self.S_dw = new_S_dw
                fast_weights = new_fast_weights

                # window-aware "future" loss, use next snapshot as target
                val_loss, avg_pr, aucnpr, auc_roc = self._step_with_fast_weights(target, fast_weights)

                if random.random() > self.drop_snap:
                    window_loss_sum = window_loss_sum + val_loss
                    window_avg_pr_list.append(avg_pr.detach())
                    window_aucnpr_list.append(aucnpr.detach())
                    window_auc_roc_list.append(auc_roc.detach())

            # outer update, use aggregated future losses for this window
            if len(window_aucnpr_list) > 0:
                window_loss = window_loss_sum / len(window_aucnpr_list)

                optimizer.zero_grad()
                self.manual_backward(window_loss)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                optimizer.step()

                last_window_loss = window_loss
                aucnpr_window_means.append(torch.stack(window_aucnpr_list).mean().item())
                avg_pr_window_means.append(torch.stack(window_avg_pr_list).mean().item())
                auc_roc_window_means.append(torch.stack(window_auc_roc_list).mean().item())
        elapsed_time = time.time() - start_time
        self.log("backprop_time", elapsed_time, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if len(aucnpr_window_means) > 0:
            mean_aucnpr = float(torch.tensor(aucnpr_window_means).mean().item())
            mean_avg_pr = float(torch.tensor(avg_pr_window_means).mean().item())
            mean_auc_roc = float(torch.tensor(auc_roc_window_means).mean().item())
            self.log("train_avg_pr", mean_avg_pr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_aucnpr", mean_aucnpr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_au_roc", mean_auc_roc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_loss", last_window_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return last_window_loss
        else:
            # no window processed (e.g., very short sequence), should we return dummy loss?
            dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
            self.log("train_aucnpr", 0.0, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_avg_pr", 0.0, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_aucnpr", 0.0, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_au_roc", 0.0, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return dummy_loss

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
