import torch
from torch import nn
from _models import register_model
from typing import List
from torch.utils.data import DataLoader
from _models._utils import BaseModel


@register_model("fedsum")
class FedSum(BaseModel):
    def __init__(
        self,
        fabric,
        network: nn.Module,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 3e-4,
        wd_reg: float = 0,
        sum_type: str = "treshold",
    ) -> None:
        self.lr = lr
        self.wd = wd_reg
        self.sum_type = sum_type

        super().__init__(fabric, network, device, optimizer, lr, wd_reg)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,task_id, update: bool = True) -> float:
        self.optimizer.zero_grad()
        with self.fabric.autocast():
            aug_inputs = self.augment(inputs)
            outputs = self.network(aug_inputs,task_id=task_id)
            loss = self.loss(outputs, labels)

        if update:
            self.fabric.backward(loss)
            self.optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            # print(f"True labels:      {labels[:10].cpu().tolist()}")
            # print(f"Predicted labels: {preds[:10].cpu().tolist()}")    

            return loss.item(),preds
        
    def end_round_client(self, dataloader: DataLoader,task):
        pass
    
    def end_round_server(self, client_info: List[dict],task):
        if len(client_info) == 0:
            return

        if self.sum_type == "treshold":
            total_samples = sum(client["num_train_samples"] for client in client_info)
            norm_weights = [
                client["num_train_samples"] / total_samples
                for client in client_info
            ]

        if self.sum_type == "treshold":
            # threshold-based aggregation
            selected = [
                (client["state_dict"], norm_weight)
                for client, norm_weight in zip(client_info, norm_weights)
                if norm_weight > 1 / (len(client_info) * 10)
            ]

            if len(selected) == 0:
                return

            agg_state = {}
            for key in selected[0][0].keys():
                agg_state[key] = sum(
                    state[key] * weight
                    for state, weight in selected
                )

            self.network.load_state_dict(agg_state, strict=True)

        else:
            # same as: torch.stack(params).sum(0)
            agg_state = {}
            for key in client_info[0]["state_dict"].keys():
                agg_state[key] = sum(
                    client["state_dict"][key] for client in client_info
                )

            self.network.load_state_dict(agg_state, strict=True)

    def begin_round_client(self, dataloader: DataLoader, server_info: dict,task):
        self.network.load_state_dict(server_info["state_dict"], strict=True)

    def get_client_info(self, dataloader: DataLoader):
        return {
            "state_dict": self.network.state_dict(),
            "num_train_samples": len(dataloader.dataset),
        }

    def get_server_info(self):
        return {"state_dict": self.network.state_dict()}
    
