from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from _models import register_model
from typing import List
from torch.utils.data import DataLoader
from _models._utils import BaseModel
from utils.tools import str_to_bool
from _networks.vit_prompt_hgp import VitHGP
from torch.optim import SGD
from _networks.vit import VisionTransformer


@register_model("ewc")
class EWC(BaseModel):

    def __init__(
        self,
        fabric,
        network: nn.Module,
        device: str,
        num_comm_rounds: int,
        optimizer: str = "AdamW",
        lr: float = 3e-4,
        wd_reg: float = 0,
        avg_type: str = "weighted",
        linear_probe: str_to_bool = False,
        slca: str_to_bool = False,
        ewc_lambda: float = 1,
        ewc_gamma: float = 0.9,
    ) -> None:
        if type(network) == VitHGP:
            for n, p in network.named_parameters():
                if "prompt" or "last" in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            params = [{"params": network.last.parameters()}, {"params": network.prompt.parameters()}]
            super().__init__(fabric, network, device, optimizer, lr, wd_reg, params=params)
        elif type(network) == VisionTransformer:
            params_backbone = []
            params_head = []
            for n, p in network.named_parameters():
                p.requires_grad = True
                if "head" not in n:
                    params_backbone.append(p)
                else:
                    params_head.append(p)
            params = [{"params": params_head}, {"params": params_backbone}]
            if slca:
                params = [{"params": params_backbone, "lr": lr / 100}, {"params": params_head}]
            super().__init__(fabric, network, device, optimizer, lr, wd_reg, params=params)
        else:
            super().__init__(fabric, network, device, optimizer, lr, wd_reg)
        self.avg_type = avg_type
        self.do_linear_probe = linear_probe
        self.done_linear_probe = False
        self.slca = slca
        self.lr = lr
        self.wd = wd_reg
        self.num_comm_rounds = num_comm_rounds
        self.ewc_lambda = ewc_lambda
        self.ewc_gamma = ewc_gamma
        self.soft = torch.nn.Softmax(dim=1)
        self.logsoft = torch.nn.LogSoftmax(dim=1)
        self.checkpoint = None
        self.optimizer_str = optimizer
        self.fish = None
        self.minibatch_size = 16
        self.round = 0

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, update: bool = True) -> float:
        if self.round == self.num_comm_rounds:
            return 0
        else:
            with self.fabric.autocast():
                inputs = self.augment(inputs)
                outputs = self.network(inputs)[:, self.cur_offset : self.cur_offset + self.cpt]
                loss = self.loss(outputs, labels % self.cpt)
                
                if self.checkpoint is not None:
                    penalty = 0.0
                    for n, p in self.network.named_parameters():
                        if n in self.fish:
                            penalty += (self.fish[n] * (p - self.checkpoint[n]) ** 2).sum()
                    loss += self.ewc_lambda * penalty
            if update:
                self.fabric.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

            return loss.item()

    def begin_task(self, n_classes_per_task: int):
        super().begin_task(n_classes_per_task)
        if self.do_linear_probe:
            self.done_linear_probe = False
        self.round = 0

    def linear_probe(self, dataloader: DataLoader):
        for epoch in range(5):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    pre_logits = self.network(inputs, pen=True, train=False)
                outputs = self.network.last(pre_logits)[:, self.cur_offset : self.cur_offset + self.cpt]
                labels = labels % self.cpt
                loss = F.cross_entropy(outputs, labels)
                self.optimizer.zero_grad()
                self.fabric.backward(loss)
                self.optimizer.step()

    def end_round_server(self, client_info: List[dict],task=0):

        total_samples = sum(c["num_train_samples"] for c in client_info)

        avg_state = {}
        for key in client_info[0]["state_dict"].keys():
            avg_state[key] = sum(
                c["state_dict"][key] * (c["num_train_samples"] / total_samples)
                for c in client_info
            )

        self.network.load_state_dict(avg_state, strict=True)
        self.total_samples = total_samples
            
    def end_task_server(self, client_info: List[dict] = None):
        super().end_task_server(client_info)
        self.checkpoint = {
            k: v.detach().clone()
            for k, v in self.network.state_dict().items()
        }

        total_samples = sum(c["num_train_samples"] for c in client_info)

        # Only update Fisher info at the last communication round
        if self.round == self.num_comm_rounds:
            # Compute weighted average of client Fishers
            print("Updating Fisher matrix")
            new_fisher = {}
            for key in client_info[0]["fisher"].keys():
                new_fisher[key] = sum(
                    c["fisher"][key] * (c["num_train_samples"] / total_samples)
                    for c in client_info
                )

            # If no previous Fisher, initialize; else decay + add new
            if self.fish is None:
                self.fish = new_fisher
            else:
                for k in self.fish:
                    self.fish[k] = self.ewc_gamma * self.fish[k] + new_fisher[k].to(self.fish[k].device)

        # Clear total_samples if needed
        self.total_samples = None

    def get_penalty_grads(self):
        return self.ewc_lambda * 2 * self.fish * (self.network.module.get_params().data - self.checkpoint)
  
    def begin_round_server(self):
        self.round += 1

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        self.round += 1
        self.network.load_state_dict(server_info["state_dict"], strict=True)

        if server_info["checkpoint"] is not None:
            self.checkpoint = {
                k: v.to(self.device)
                for k, v in server_info["checkpoint"].items()
            }
        if server_info["fisher"] is not None:
            self.fish = {
                k: v.to(self.device)
                for k, v in server_info["fisher"].items()
            }
        if self.do_linear_probe and not self.done_linear_probe:
            optimizer = self.optimizer_class(self.network.last.parameters(), lr=self.lr, weight_decay=self.wd)
            self.optimizer = self.fabric.setup_optimizers(optimizer)
            self.linear_probe(dataloader)
            self.done_linear_probe = True
            # restore correct optimizer
            params = [{"params": self.network.last.parameters()}, {"params": self.network.prompt.parameters()}]
            optimizer = self.optimizer_class(params, lr=self.lr, weight_decay=self.wd)
            self.optimizer = self.fabric.setup_optimizers(optimizer)
        
        OptimizerClass = getattr(torch.optim, self.optimizer_str)
        self.optimizer_class = OptimizerClass
        self.optimizer = OptimizerClass(self.network.parameters(), lr=self.lr, weight_decay=self.wd)
        self.optimizer = self.fabric.setup_optimizers(self.optimizer)

    def get_client_info(self, dataloader):

        return {
            "state_dict": {
                k: v.detach().cpu()
                for k, v in self.network.state_dict().items()
            },
            "num_train_samples": len(dataloader.dataset),
            "fisher": {
                k: v.detach().cpu()
                for k, v in self.fish.items()
            } if self.fish is not None else None,
        }
    
    def end_task_client(self, train_loader, server_info):
        fish = {n: torch.zeros_like(p)
                for n, p in self.network.module.named_parameters()
                if p.requires_grad}
        fake_opt = SGD(self.network.module.parameters(), lr=0)
        for j, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device).long()
            inputs = self.test_transform(inputs)
            for ex, lab in zip(inputs, labels):
                output = self.network.module(ex.unsqueeze(0))[:, self.cur_offset : self.cur_offset + self.cpt]
                loss = - F.nll_loss(self.logsoft(output), lab.unsqueeze(0) % self.cpt,
                                    reduction='none')
                exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                loss = torch.mean(loss)
                loss.backward()
                for n, p in self.network.module.named_parameters():
                    if p.requires_grad and p.grad is not None and n in fish:
                        fish[n] += exp_cond_prob * p.grad ** 2
                fake_opt.zero_grad()

        fake_opt = None

        self.fish = {n: v.cpu() for n, v in fish.items()}
        return self.get_client_info(train_loader)

    def end_round_client(self, dataloader: DataLoader):
        super().end_round_client(dataloader)
        self.optimizer = None
        self.checkpoint = None
        
    def get_server_info(self):
        return {
            "state_dict": {
                k: v.detach().cpu()
                for k, v in self.network.state_dict().items()
            },
            "checkpoint": (
                {k: v.detach().cpu() for k, v in self.checkpoint.items()}
                if self.checkpoint is not None
                else None
            ),
            "fisher": (
                {k: v.detach().cpu() for k, v in self.fish.items()}
                if self.fish is not None
                else None
            ),
        }    



    def to(self, device):
        self.network.to(device)

        if self.checkpoint is not None:
            self.checkpoint = {
                k: v.to(device)
                for k, v in self.checkpoint.items()
            }

        if self.fish is not None:
            self.fish = {
                k: v.to(device)
                for k, v in self.fish.items()
            }

        return self