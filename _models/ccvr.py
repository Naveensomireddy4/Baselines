# import torch
# from torch import nn
# from torch.nn import functional as F
# from torch.distributions import MultivariateNormal
# from _models import register_model
# from typing import List
# from torch.utils.data import DataLoader
# from _models._utils import BaseModel
# from _networks.vit import VisionTransformer as Vit
# import os
# from utils.tools import str_to_bool


# @register_model("ccvr")
# class CCVR(BaseModel):
#     def __init__(
#         self,
#         fabric,
#         network: Vit,
#         device: str,
#         optimizer: str = "AdamW",
#         lr: float = 1e-3,
#         wd_reg: float = 0,
#         avg_type: str = "weighted",
#         how_many: int = 256,
#         full_cov: str_to_bool = False,
#         linear_probe: str_to_bool = False,
#     ) -> None:
#         params = [{"params": network.parameters()}]
#         super().__init__(fabric, network, device, optimizer, lr, wd_reg, params=params)
#         self.avg_type = avg_type
#         self.how_many = how_many
#         self.clients_statistics = None
#         self.mogs = {}
#         self.logit_norm = 0.1
#         self.full_cov = full_cov
#         self.do_linear_probe = linear_probe
#         self.done_linear_probe = False
#         self.lr = lr
#         self.wd_reg = wd_reg
#         self.cpt = []

#     def observe(self, inputs: torch.Tensor, labels: torch.Tensor, update: bool = True) -> float:
#         self.optimizer.zero_grad()
#         with self.fabric.autocast():
#             inputs = self.augment(inputs)
#             outputs = self.network(inputs)[:, self.cur_offset : self.cur_offset + self.cpt[-1]]
#             loss = self.loss(outputs, labels - self.cur_offset)

#         if update:
#             self.fabric.backward(loss)
#             # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
#             self.optimizer.step()

#         return loss.item()

#     def linear_probe(self, dataloader: DataLoader):
#         for epoch in range(5):
#             for inputs, labels in dataloader:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
#                 with torch.no_grad():
#                     pre_logits = self.network(inputs, pen=True, train=False)
#                 outputs = self.network(pre_logits)[:, self.cur_offset : self.cur_offset + self.cpt[-1]]
#                 labels = labels - self.cur_offset
#                 loss = F.cross_entropy(outputs, labels)
#                 self.optimizer.zero_grad()
#                 self.fabric.backward(loss)
#                 self.optimizer.step()

#     def begin_task(self, n_classes_per_task: int):
#         #super().begin_task(n_classes_per_task)
#         self.cur_task += 1
#         if self.cur_task > 0:
#             self.cur_offset += self.cpt[-1]
#         self.cpt.append(n_classes_per_task)
#         if self.do_linear_probe:
#             self.done_linear_probe = False

#     def end_round_server(self, client_info: List[dict]):
#         if self.avg_type == "weighted":
#             total_samples = sum([client["num_train_samples"] for client in client_info])
#             norm_weights = [client["num_train_samples"] / total_samples for client in client_info]
#         else:
#             weights = [1 if client["num_train_samples"] > 0 else 0 for client in client_info]
#             norm_weights = [w / sum(weights) for w in weights]

#         if len(client_info) > 0:
#             avg_state = {}
#             for key in client_info[0]["state_dict"].keys():
#                 avg_state[key] = sum(
#                     c["state_dict"][key] * (c["num_train_samples"] / total_samples)
#                     for c in client_info
#                 )

#             self.network.load_state_dict(avg_state, strict=True)
#         clients_gaussians = [client["client_statistics"] for client in client_info]
#         self.to(self.device)
#         mogs = {}
#         for clas in range(self.cur_offset, self.cur_offset + self.cpt[-1]):
#             counter = 0
#             for client_gaussians in clients_gaussians:
#                 if client_gaussians.get(clas) is not None:
#                     gaus_data = []
#                     gaus_mean = client_gaussians[clas][1]
#                     gaus_var = client_gaussians[clas][2]
#                     gaus_data.append(gaus_mean)
#                     gaus_data.append(gaus_var)
#                     weight = client_gaussians[clas][0]
#                     if mogs.get(clas) is None:
#                         mogs[clas] = [[weight], [gaus_mean], [gaus_var]]
#                     else:
#                         mogs[clas][0].append(weight)
#                         mogs[clas][1].append(gaus_mean)
#                         mogs[clas][2].append(gaus_var)
#                     counter += client_gaussians[clas][0]
#             if mogs.get(clas) is not None:
#                 mogs[clas][0] = [mogs[clas][0][i] / counter for i in range(len(mogs[clas][0]))]
#         self.mogs = mogs
#         if "t5" not in str(type(self.network)).lower():
#             optimizer = torch.optim.SGD(self.network.parameters(), lr=0.01, momentum=0.9, weight_decay=0)
#         else:
#             optimizer = torch.optim.SGD(self.network.parameters(), lr=0.01, momentum=0.9, weight_decay=0)
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=5)
#         logits_norm = torch.tensor([], dtype=torch.float32).to(self.device)
#         for epoch in range(5):
#             sampled_data = []
#             sampled_label = []
#             # TODO: fix the probabilities of the classes
#             # since Cifar100 and Tiny-ImageNet are balanced datasets and the participation rate is 100%,
#             # we can set classes_weights asequiprobable
#             num_cur_classes = self.cpt[-1]
#             classes_weights = torch.ones(num_cur_classes, dtype=torch.float32).to(self.device)
#             classes_samples = torch.multinomial(classes_weights, self.how_many * num_cur_classes, replacement=True)
#             _, classes_samples = torch.unique(classes_samples, return_counts=True)
#             # sample features from gaussians:
#             for clas in range(self.cur_offset, self.cur_offset + self.cpt[-1]):
#                 if self.mogs.get([clas][0]) is None:
#                     #print("No gaussian for class ", task * self.cpt + clas)
#                     continue
#                 weights_list = []
#                 for weight in self.mogs[clas][0]:
#                     weights_list.append(weight)
#                 gaussian_samples = torch.zeros(len(weights_list), dtype=torch.int64).to(self.device)
#                 weights_list = torch.tensor(weights_list, dtype=torch.float32).to(self.device)
#                 gaussian_samples_fill = torch.multinomial(
#                         weights_list, classes_samples[clas - self.cur_offset], replacement=True
#                 )
#                 gaussian_clients, gaussian_samples_fill = torch.unique(gaussian_samples_fill, return_counts=True)
#                 gaussian_samples[gaussian_clients] += gaussian_samples_fill
#                 for id, (mean, variance) in enumerate(
#                     zip(
#                         self.mogs[clas][1],
#                         self.mogs[clas][2],
#                     )
#                 ):
#                     cls_mean = mean  # * (0.9 + decay)
#                     cls_var = variance
#                     if self.full_cov:
#                         cov = cls_var + 1e-8 * torch.eye(cls_mean.shape[-1]).to(self.device)
#                     else:
#                         cov = (torch.eye(cls_mean.shape[-1]).to(self.device) * cls_var) + (1e-8 * torch.eye(cls_mean.shape[-1]).to(self.device))
#                     m = MultivariateNormal(cls_mean, cov)
#                     n_samples = int(torch.round(gaussian_samples[id]))
#                     sampled_data_single = m.sample((n_samples,))
#                     sampled_data.append(sampled_data_single)
#                     sampled_label.extend([clas] * n_samples)
#             sampled_data = torch.cat(sampled_data, 0).float().to(self.device)
#             sampled_label = torch.tensor(sampled_label, dtype=torch.int64).to(self.device)
#             inputs = sampled_data
#             targets = sampled_label

#             sf_indexes = torch.randperm(inputs.size(0))
#             inputs = inputs[sf_indexes]
#             targets = targets[sf_indexes]
#             for _iter in range(self.cpt[-1]):
#                 inp = inputs[_iter * self.how_many : (_iter + 1) * self.how_many].to(self.device)
#                 tgt = targets[_iter * self.how_many : (_iter + 1) * self.how_many].to(self.device)
#                 if "t5" not in str(type(self.network)).lower():
#                     outputs = self.network(inp)
#                 else:
#                     outputs = self.network(inp)
#                 logits = outputs[:, self.cur_offset:self.cur_offset+self.cpt[-1]]
#                 temp_norm = torch.norm(logits, p=2, dim=-1, keepdim=True)
#                 norms = temp_norm.mean(dim=-1, keepdim=True)

#                 decoupled_logits = torch.div(logits + 1e-12, norms + 1e-12) / self.logit_norm
#                 loss = F.cross_entropy(decoupled_logits, tgt - self.cur_offset)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#             scheduler.step()

#     def begin_round_client(self, dataloader: DataLoader, server_info: dict):
#         self.network.load_state_dict(server_info["state_dict"], strict=True)
#         if self.do_linear_probe and not self.done_linear_probe:
#             optimizer = self.optimizer_class(self.network.parameters(), lr=self.lr, weight_decay=self.wd_reg)
#             self.optimizer = self.fabric.setup_optimizers(optimizer)
#             self.linear_probe(dataloader)
#             self.done_linear_probe = True
#         # restore correct optimizer
#         params = [{"params": self.network.parameters()}]
#         optimizer = self.optimizer_class(params, lr=self.lr, weight_decay=self.wd_reg)
#         self.optimizer = self.fabric.setup_optimizers(optimizer)

#     def get_client_info(self, dataloader: DataLoader):
#         return {
#             "state_dict": self.network.state_dict(),
#             "num_train_samples": len(dataloader.dataset.data),
#             "client_statistics": self.clients_statistics,
#         }

#     def get_server_info(self):
#         return {"state_dict": self.network.state_dict()}

#     def end_round_client(self, dataloader: DataLoader):
#         features = torch.tensor([], dtype=torch.float32).to(self.device)
#         true_labels = torch.tensor([], dtype=torch.int64).to(self.device)
#         num_epochs = 1 if not self.full_cov else 3
#         with torch.no_grad():
#             client_statistics = {}
#             for _ in range(num_epochs):
#                 for id, data in enumerate(dataloader):
#                     inputs, labels = data
#                     inputs, labels = inputs.to(self.device), labels.to(self.device)
#                     outputs = self.network(inputs, returnt="features")
#                     features = torch.cat((features, outputs), 0)
#                     true_labels = torch.cat((true_labels, labels), 0)
#             client_labels = torch.unique(true_labels).tolist()
#             for client_label in client_labels:
#                 number = (true_labels == client_label).sum().item()
#                 if number > 1:
#                     gaussians = []
#                     gaussians.append(number)
#                     gaussians.append(torch.mean(features[true_labels == client_label], 0))
#                     if self.full_cov:
#                         gaussians.append(
#                             torch.cov(features[true_labels == client_label].T.type(torch.float64))
#                             .type(torch.float32)
#                             .to(self.device)
#                         )
#                     else:
#                         gaussians.append(torch.std(features[true_labels == client_label], 0) ** 2)
#                     client_statistics[client_label] = gaussians
#             self.clients_statistics = client_statistics

#     def save_checkpoint(self, output_folder: str, task: int, comm_round: int) -> None:
#         training_status = self.network.training
#         self.network.eval()

#         checkpoint = {
#             "task": task,
#             "comm_round": comm_round,
#             "network": self.network,
#             "optimizer": self.optimizer,
#             "mogs": self.mogs,
#         }
#         name = "hgp_" + "full_cov" if self.full_cov else "diag_cov"
#         name += "_linear_probe" if self.do_linear_probe else ""
#         name += f"_task_{task}_round_{comm_round}_checkpoint.pt"
#         self.fabric.save(os.path.join(output_folder, name), checkpoint)
#         self.network.train(training_status)



import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
from _models import register_model
from typing import List
from torch.utils.data import DataLoader
from _models._utils import BaseModel
import os
from utils.tools import str_to_bool


@register_model("ccvr")
class CCVR(BaseModel):

    def __init__(
        self,
        fabric,
        network,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 1e-3,
        wd_reg: float = 0,
        avg_type: str = "weighted",
        how_many: int = 500,
        full_cov: str_to_bool = False,
        linear_probe: str_to_bool = False,
    ) -> None:

        params = [{"params": network.parameters()}]

        super().__init__(
            fabric,
            network,
            device,
            optimizer,
            lr,
            wd_reg,
            params=params
        )

        self.avg_type = avg_type
        self.how_many = how_many
        self.clients_statistics = None
        self.mogs = {}
        self.logit_norm = 0.1
        self.full_cov = full_cov

        self.do_linear_probe = linear_probe
        self.done_linear_probe = False

        self.lr = lr
        self.wd_reg = wd_reg

        self.cpt = []

    ###########################################################################
    # TRAIN
    ###########################################################################

    def observe(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        update: bool = True
    ) -> float:

        self.optimizer.zero_grad()

        with self.fabric.autocast():

            inputs = self.augment(inputs)

            outputs = self.network(inputs)[
                :,
                self.cur_offset:self.cur_offset + self.cpt[-1]
            ]

            loss = self.loss(outputs, labels - self.cur_offset)

        if update:

            self.fabric.backward(loss)

            self.optimizer.step()

        return loss.item()

    ###########################################################################
    # LINEAR PROBE
    ###########################################################################

    def linear_probe(self, dataloader: DataLoader):

        for epoch in range(5):

            for inputs, labels in dataloader:

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                with torch.no_grad():

                    features = self.network(
                        inputs,
                        returnt="features"
                    )

                if hasattr(self.network, "module"):
                    outputs = self.network.module.classifier(features)
                else:
                    outputs = self.network.classifier(features)

                outputs = outputs[
                    :,
                    self.cur_offset:self.cur_offset + self.cpt[-1]
                ]

                labels = labels - self.cur_offset

                loss = F.cross_entropy(outputs, labels)

                self.optimizer.zero_grad()

                self.fabric.backward(loss)

                self.optimizer.step()

    ###########################################################################
    # TASK
    ###########################################################################

    def begin_task(self, n_classes_per_task: int):

        self.cur_task += 1

        if self.cur_task > 0:
            self.cur_offset += self.cpt[-1]

        self.cpt.append(n_classes_per_task)

        if self.do_linear_probe:
            self.done_linear_probe = False

    ###########################################################################
    # SERVER
    ###########################################################################

    def end_round_server(self, client_info: List[dict]):

        #######################################################################
        # FEDAVG
        #######################################################################

        if self.avg_type == "weighted":

            total_samples = sum(
                [client["num_train_samples"] for client in client_info]
            )

        else:

            total_samples = len(client_info)

        if len(client_info) > 0:

            avg_state = {}

            for key in client_info[0]["state_dict"].keys():

                avg_state[key] = sum(
                    c["state_dict"][key]
                    * (
                        c["num_train_samples"]
                        / total_samples
                    )
                    for c in client_info
                )

            if hasattr(self.network, "module"):
                self.network.module.load_state_dict(
                    avg_state,
                    strict=True
                )
            else:
                self.network.load_state_dict(
                    avg_state,
                    strict=True
                )

        #######################################################################
        # BUILD MIXTURE OF GAUSSIANS
        #######################################################################

        clients_gaussians = [
            client["client_statistics"]
            for client in client_info
        ]

        self.to(self.device)

        mogs = {}

        for clas in range(
            self.cur_offset,
            self.cur_offset + self.cpt[-1]
        ):

            counter = 0

            for client_gaussians in clients_gaussians:

                if client_gaussians.get(clas) is not None:

                    gaus_mean = client_gaussians[clas][1]
                    gaus_var = client_gaussians[clas][2]

                    weight = client_gaussians[clas][0]

                    if mogs.get(clas) is None:

                        mogs[clas] = [
                            [weight],
                            [gaus_mean],
                            [gaus_var]
                        ]

                    else:

                        mogs[clas][0].append(weight)
                        mogs[clas][1].append(gaus_mean)
                        mogs[clas][2].append(gaus_var)

                    counter += weight

            if mogs.get(clas) is not None:

                mogs[clas][0] = [
                    mogs[clas][0][i] / counter
                    for i in range(len(mogs[clas][0]))
                ]

        self.mogs = mogs

        #######################################################################
        # REPLAY OPTIMIZER
        #######################################################################

        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=5
        )

        #######################################################################
        # FEATURE REPLAY
        #######################################################################

        for epoch in range(5):

            sampled_data = []
            sampled_label = []

            num_cur_classes = self.cpt[-1]

            classes_samples = torch.full(
                (num_cur_classes,),
                self.how_many,
                dtype=torch.int64,
                device=self.device
            )

            ###################################################################
            # SAMPLE FEATURES
            ###################################################################

            for clas in range(
                self.cur_offset,
                self.cur_offset + self.cpt[-1]
            ):

                if self.mogs.get(clas) is None:
                    continue

                weights_list = torch.tensor(
                    self.mogs[clas][0],
                    dtype=torch.float32,
                    device=self.device
                )

                gaussian_samples = torch.zeros(
                    len(weights_list),
                    dtype=torch.int64,
                    device=self.device
                )

                gaussian_samples_fill = torch.multinomial(
                    weights_list,
                    classes_samples[clas - self.cur_offset],
                    replacement=True
                )

                gaussian_clients, gaussian_samples_fill = torch.unique(
                    gaussian_samples_fill,
                    return_counts=True
                )

                gaussian_samples[
                    gaussian_clients
                ] += gaussian_samples_fill

                ################################################################
                # SAMPLE EACH GAUSSIAN
                ################################################################

                for idx, (mean, variance) in enumerate(
                    zip(
                        self.mogs[clas][1],
                        self.mogs[clas][2]
                    )
                ):

                    cls_mean = mean
                    cls_var = variance

                    if self.full_cov:

                        cov = (
                            cls_var
                            + 1e-8
                            * torch.eye(
                                cls_mean.shape[-1]
                            ).to(self.device)
                        )

                    else:

                        cov = torch.diag(cls_var + 1e-8)

                    m = MultivariateNormal(
                        cls_mean,
                        cov
                    )

                    n_samples = int(
                        torch.round(
                            gaussian_samples[idx].float()
                        ).item()
                    )

                    if n_samples <= 0:
                        continue

                    sampled_data_single = m.sample(
                        (n_samples,)
                    )

                    sampled_data.append(
                        sampled_data_single
                    )

                    sampled_label.extend(
                        [clas] * n_samples
                    )

            ###################################################################
            # EMPTY CHECK
            ###################################################################

            if len(sampled_data) == 0:
                continue

            sampled_data = torch.cat(
                sampled_data,
                0
            ).float().to(self.device)

            sampled_label = torch.tensor(
                sampled_label,
                dtype=torch.int64
            ).to(self.device)

            ###################################################################
            # SHUFFLE
            ###################################################################

            sf_indexes = torch.randperm(
                sampled_data.size(0)
            )

            inputs = sampled_data[sf_indexes]
            targets = sampled_label[sf_indexes]

            ###################################################################
            # TRAIN CLASSIFIER
            ###################################################################

            for _iter in range(self.cpt[-1]):

                inp = inputs[
                    _iter * self.how_many:
                    (_iter + 1) * self.how_many
                ]

                tgt = targets[
                    _iter * self.how_many:
                    (_iter + 1) * self.how_many
                ]

                ###############################################################
                # IMPORTANT FIX
                ###############################################################

                if inp.dim() == 2:

                    if hasattr(self.network, "module"):
                        outputs = self.network.module.classifier(inp)
                    else:
                        outputs = self.network.classifier(inp)

                else:

                    outputs = self.network(inp)

                ###############################################################

                logits = outputs[
                    :,
                    self.cur_offset:
                    self.cur_offset + self.cpt[-1]
                ]

                temp_norm = torch.norm(
                    logits,
                    p=2,
                    dim=-1,
                    keepdim=True
                )

                norms = temp_norm.mean(
                    dim=-1,
                    keepdim=True
                )

                decoupled_logits = (
                    torch.div(
                        logits + 1e-12,
                        norms + 1e-12
                    )
                    / self.logit_norm
                )

                loss = F.cross_entropy(
                    decoupled_logits,
                    tgt - self.cur_offset
                )

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            scheduler.step()

    ###########################################################################
    # CLIENT
    ###########################################################################

    def begin_round_client(
        self,
        dataloader: DataLoader,
        server_info: dict
    ):

        self.network.load_state_dict(
            server_info["state_dict"],
            strict=True
        )

        if (
            self.do_linear_probe
            and not self.done_linear_probe
        ):

            optimizer = self.optimizer_class(
                self.network.parameters(),
                lr=self.lr,
                weight_decay=self.wd_reg
            )

            self.optimizer = self.fabric.setup_optimizers(
                optimizer
            )

            self.linear_probe(dataloader)

            self.done_linear_probe = True

        params = [{"params": self.network.parameters()}]

        optimizer = self.optimizer_class(
            params,
            lr=self.lr,
            weight_decay=self.wd_reg
        )

        self.optimizer = self.fabric.setup_optimizers(
            optimizer
        )

    ###########################################################################
    # INFO
    ###########################################################################

    def get_client_info(self, dataloader: DataLoader):

        return {
            "state_dict": {
                k: v.detach().cpu()
                for k, v in self.network.state_dict().items()
            },

            "num_train_samples": len(dataloader.dataset),

            "client_statistics": self.clients_statistics,
        }

    def get_server_info(self):

        return {
            "state_dict": {
                k: v.detach().cpu()
                for k, v in self.network.state_dict().items()
            }
        }

    ###########################################################################
    # CLIENT STATISTICS
    ###########################################################################

    def end_round_client(
        self,
        dataloader: DataLoader
    ):

        features = torch.tensor(
            [],
            dtype=torch.float32
        ).to(self.device)

        true_labels = torch.tensor(
            [],
            dtype=torch.int64
        ).to(self.device)

        num_epochs = 1 if not self.full_cov else 3

        with torch.no_grad():

            client_statistics = {}

            for _ in range(num_epochs):

                for _, data in enumerate(dataloader):

                    inputs, labels = data

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.network(
                        inputs,
                        returnt="features"
                    )

                    features = torch.cat(
                        (features, outputs),
                        0
                    )

                    true_labels = torch.cat(
                        (true_labels, labels),
                        0
                    )

            client_labels = torch.unique(
                true_labels
            ).tolist()

            for client_label in client_labels:

                number = (
                    true_labels == client_label
                ).sum().item()

                if number > 1:

                    gaussians = []

                    gaussians.append(number)

                    gaussians.append(
                        torch.mean(
                            features[
                                true_labels == client_label
                            ],
                            0
                        )
                    )

                    if self.full_cov:

                        gaussians.append(
                            torch.cov(
                                features[
                                    true_labels == client_label
                                ].T.type(torch.float64)
                            )
                            .type(torch.float32)
                            .to(self.device)
                        )

                    else:

                        gaussians.append(
                            torch.var(
                                features[
                                    true_labels == client_label
                                ],
                                dim=0,
                                unbiased=False
                            ) + 1e-8
                        )

                    client_statistics[
                        client_label
                    ] = gaussians

            self.clients_statistics = client_statistics

    ###########################################################################
    # SAVE
    ###########################################################################

    def save_checkpoint(
        self,
        output_folder: str,
        task: int,
        comm_round: int
    ) -> None:

        training_status = self.network.training

        self.network.eval()

        checkpoint = {
            "task": task,
            "comm_round": comm_round,
            "network": self.network,
            "optimizer": self.optimizer,
            "mogs": self.mogs,
        }

        name = (
            "hgp_"
            + (
                "full_cov"
                if self.full_cov
                else "diag_cov"
            )
        )

        name += (
            "_linear_probe"
            if self.do_linear_probe
            else ""
        )

        name += (
            f"_task_{task}"
            f"_round_{comm_round}"
            "_checkpoint.pt"
        )

        self.fabric.save(
            os.path.join(output_folder, name),
            checkpoint
        )

        self.network.train(training_status)