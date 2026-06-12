from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import BatchEncoding
from kornia import augmentation as K


class BaseDataset:
    NAME = None
    N_CLASSES_PER_TASK = None
    N_TASKS = None
    TRAIN_TRANSFORM = None
    TEST_TRANSFORM = None
    IS_TEXT = False

    def __init__(
        self,
        num_clients,
        batch_size,
        partition_mode,
        distribution_alpha,
        class_quantity,
    ):
        self.train_data, self.train_targets = [], []
        self.test_data, self.test_targets = [], []
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.train_transf = None
        self.test_transf = None

    def set_transforms(self, train_transform: str = None, test_transform: str = None):
        if train_transform is not None:
            self.train_transf = train_transform
        if test_transform is not None:
            self.test_transf = test_transform

    def _split_fcil_OOS(
        self,
        num_clients,
        partition_mode,
        distribution_alpha=None,
        class_quantity=None,
        format="numpy",
    ):
        assert partition_mode in ["distribution", "quantity", "extended"]
        if partition_mode == "distribution" or partition_mode == "extended":
            assert distribution_alpha is not None
        elif partition_mode == "quantity":
            assert class_quantity is not None

        num_samples_per_client = []
        for split in ["train", "test"]:
            print(f"Splitting {split} data")
            dataset = getattr(self, f"{split}_dataset")
            min_samples_split = 6 if split == "train" else 1
            for task in range(0, self.N_TASKS):
                min_samples = 0
                iterations = 0
                if split == "train" and partition_mode == "extended":
                    default_samples = 7
                    base_class = task * self.N_CLASSES_PER_TASK
                    cur_classes = np.arange(base_class, base_class + self.N_CLASSES_PER_TASK)
                    cpt = self.N_CLASSES_PER_TASK
                    total_samples = np.stack(
                        [dataset.data[dataset.targets == clas].shape[0] for clas in cur_classes]
                    ).sum()
                    classes_data = [dataset.data[dataset.targets == clas] for clas in cur_classes]
                    classes_targets = [dataset.targets[dataset.targets == clas] for clas in cur_classes]
                    unrolled_assignments_per_class = np.concatenate(
                        [np.ones(len(classes_data[clas % cpt]), dtype=int) * (-1) for clas in cur_classes]
                    ).flatten()
                    clients_assignments_per_class = [
                        np.ones(len(classes_data[clas % cpt]), dtype=int) * (-1) for clas in cur_classes
                    ]
                    clients_classes_distr = np.random.dirichlet(
                        np.repeat(0.05, num_clients), size=len(cur_classes)
                    )
                    classes_clients_numbers = {clas % cpt: [] for clas in cur_classes}
                    for clas in cur_classes:
                        classes_clients_numbers[clas % cpt].append([c for c in range(num_clients)])
                        classes_clients_numbers[clas % cpt].append(np.zeros((num_clients,), dtype=int))
                    for client in range(num_clients):
                        distr = torch.distributions.Categorical(torch.tensor(clients_classes_distr[:, client]))
                        classes_to_sample = distr.sample((default_samples,)).numpy()
                        for clas in classes_to_sample:
                            classes_clients_numbers[clas % cpt][1][client] += 1
                    for clas in cur_classes:
                        max_value = clients_assignments_per_class[clas % cpt].shape[0]
                        samples = np.random.choice(max_value, sum(classes_clients_numbers[clas % cpt][1]), replace=True)
                        tmp = 0
                        for client in range(num_clients):
                            how_many_samples = classes_clients_numbers[clas % cpt][1][client]
                            if how_many_samples != 0:
                                clients_assignments_per_class[clas % cpt][
                                    samples[tmp: tmp + how_many_samples]
                                ] = client
                                tmp += how_many_samples

                while min_samples < min_samples_split:
                    task_data_sentences = [list() for _ in range(num_clients)]
                    task_data_attmat = [list() for _ in range(num_clients)]
                    task_targets = [list() for _ in range(num_clients)]
                    if isinstance(self.N_CLASSES_PER_TASK, list):
                        base_class = sum(self.N_CLASSES_PER_TASK[:task])
                        cur_classes = np.arange(base_class, base_class + self.N_CLASSES_PER_TASK[task])
                    else:
                        base_class = task * self.N_CLASSES_PER_TASK
                        cur_classes = np.arange(base_class, base_class + self.N_CLASSES_PER_TASK)

                    num_samples_per_client_task = []

                    iterations += 1
                    assert iterations < 10000
                    if partition_mode == "quantity":
                        trials = 0
                        while trials < 1000:
                            trials += 1
                            clients_per_class = {cls: list() for cls in cur_classes}
                            classes_set = set()
                            for client_idx in range(num_clients):
                                assert class_quantity <= len(cur_classes)
                                chosen_classes = np.random.choice(cur_classes, class_quantity, replace=False)
                                classes_set = set(list(classes_set) + list(chosen_classes))
                                for chosen_class in chosen_classes:
                                    clients_per_class[chosen_class].append(client_idx)
                            if classes_set == set(cur_classes):
                                break
                        assert trials != 1000

                    for clas in cur_classes:
                        selected_tokenized_sentences = dataset.data["input_ids"][dataset.targets == clas]
                        selected_attention_masks = dataset.data["attention_mask"][dataset.targets == clas]
                        selected_data = {
                            "input_ids": selected_tokenized_sentences,
                            "attention_mask": selected_attention_masks,
                        }
                        class_data = BatchEncoding(data=selected_data)
                        class_targets = dataset.targets[dataset.targets == clas]
                        num_samples = len(selected_tokenized_sentences)

                        if split == "train":
                            if partition_mode == "distribution" or partition_mode == "extended":
                                probs = np.random.dirichlet(np.repeat(distribution_alpha, num_clients))
                                while np.isnan(probs).sum() > 0:
                                    probs = np.random.dirichlet(np.repeat(distribution_alpha, num_clients))
                            elif partition_mode == "quantity":
                                probs = np.zeros((num_clients,))
                                for client_idx in clients_per_class[clas]:
                                    probs[client_idx] = 1 / len(clients_per_class[clas])

                            probs2 = np.where(probs < 1e-20, 0, probs)
                            probs3 = np.clip(probs2, 0, 1)
                            client_distr = torch.distributions.Categorical(torch.tensor(probs3))
                            assigned_client = client_distr.sample((num_samples,)).numpy()
                            if partition_mode == "extended":
                                assigned_client = np.where(
                                    clients_assignments_per_class[clas % cpt] != -1,
                                    clients_assignments_per_class[clas % cpt],
                                    assigned_client,
                                )
                            num_samples_per_client_task.append(
                                [(assigned_client == i).sum().item() for i in range(num_clients)]
                            )
                        else:
                            if max(num_samples_per_client[clas]) < sum(num_samples_per_client[clas]) / num_samples:
                                train_test_ratio = 1
                            else:
                                train_test_ratio = sum(num_samples_per_client[clas]) / num_samples

                            num_samples_per_client[clas] = [
                                int(round(num_samples_per_client[clas][client_idx] / train_test_ratio))
                                for client_idx in range(num_clients)
                            ]
                            assigned_client = np.concatenate(
                                [np.ones((num_samples_per_client[clas][i],), dtype=int) * i for i in range(num_clients)]
                            )
                            assigned_client = assigned_client[np.random.permutation(assigned_client.shape[0])]

                            value, counts = np.unique(assigned_client, return_counts=True)
                            sample = value[torch.distributions.Categorical(torch.tensor(counts)).sample((num_clients,))]
                            sample = np.array(sample).reshape(-1)
                            assigned_client = np.concatenate([assigned_client, sample])
                            assigned_client = assigned_client[:num_samples]

                        for client_idx in range(num_clients):
                            sentences = class_data["input_ids"]
                            attmat = class_data["attention_mask"]
                            task_data_sentences[client_idx] += [sentences[assigned_client == client_idx]]
                            task_data_attmat[client_idx] += [attmat[assigned_client == client_idx]]
                            task_targets[client_idx] += [class_targets[assigned_client == client_idx]]

                    task_data_sentences = [
                        torch.cat([clas_data for clas_data in client_data]) for client_data in task_data_sentences
                    ]
                    task_data_attmat = [
                        torch.cat([clas_data for clas_data in client_data]) for client_data in task_data_attmat
                    ]
                    task_targets = [
                        np.concatenate([clas_data for clas_data in client_data]) for client_data in task_targets
                    ]

                    min_samples = min([len(client_data) for client_data in task_data_attmat])

                selected_data = {"input_ids": task_data_sentences, "attention_mask": task_data_attmat}
                task_data = BatchEncoding(data=selected_data)

                for i in range(len(num_samples_per_client_task)):
                    num_samples_per_client.append(num_samples_per_client_task[i])
                if format == "pytorch":
                    getattr(self, f"{split}_data").append([torch.tensor(td) for td in task_data])
                    getattr(self, f"{split}_targets").append([torch.tensor(tt) for tt in task_targets])
                else:
                    getattr(self, f"{split}_data").append(task_data)
                    getattr(self, f"{split}_targets").append(task_targets)

        print("Data split done")

    def _split_fcil(
        self,
        num_clients,
        partition_mode,
        distribution_alpha=None,
        class_quantity=None,
        format="numpy",
    ):
        assert partition_mode in ["distribution", "quantity", "extended"]

        if partition_mode in ["distribution", "extended"]:
            assert distribution_alpha is not None
        if partition_mode == "quantity":
            assert class_quantity is not None

        self.train_data, self.train_targets = [], []
        self.test_data, self.test_targets = [], []

        for split in ["train", "test"]:
            print(f"\n==================== Splitting {split.upper()} ====================")
            dataset = getattr(self, f"{split}_dataset")
            min_samples_split = 6 if split == "train" else 1

            for task in range(self.N_TASKS):
                print(f"\n--- Task {task} ---")
                min_samples = 0
                iterations = 0

                while min_samples < min_samples_split:
                    iterations += 1
                    assert iterations < 10000, "Infinite loop detected"

                    task_data    = [None for _ in range(num_clients)]
                    task_targets = [None for _ in range(num_clients)]

                    base_class  = task * self.N_CLASSES_PER_TASK
                    cur_classes = np.arange(base_class, base_class + self.N_CLASSES_PER_TASK)

                    # quantity mode
                    if partition_mode == "quantity":
                        clients_per_class = {cls: [] for cls in cur_classes}
                        for client in range(num_clients):
                            chosen = np.random.choice(cur_classes, class_quantity, replace=False)
                            for cls in chosen:
                                clients_per_class[cls].append(client)

                    client_data    = [[] for _ in range(num_clients)]
                    client_targets = [[] for _ in range(num_clients)]

                    for cls in cur_classes:
                        class_data    = dataset.data[dataset.targets == cls]
                        class_targets = dataset.targets[dataset.targets == cls]
                        # class_targets = class_targets % self.N_CLASSES_PER_TASK
                        num_samples   = len(class_data)

                        # ------------------------------
                        # SAFE DIRICHLET
                        # ------------------------------
                        if partition_mode in ["distribution", "extended"]:
                            proportions = np.random.dirichlet(
                                np.repeat(distribution_alpha, num_clients)
                            )

                            # fix instability
                            if np.any(np.isnan(proportions)) or np.any(proportions < 0):
                                proportions = np.ones(num_clients) / num_clients

                            proportions = proportions / proportions.sum()

                        else:  # quantity
                            proportions = np.zeros(num_clients)
                            for c in clients_per_class[cls]:
                                proportions[c] = 1.0 / len(clients_per_class[cls])

                        # ------------------------------
                        # SAFE COUNTS
                        # ------------------------------
                        counts = np.floor(proportions * num_samples).astype(int)

                        remainder = int(num_samples - counts.sum())
                        for i in range(remainder):
                            counts[i % num_clients] += 1

                        # ------------------------------
                        # SHUFFLE
                        # ------------------------------
                        perm = np.random.permutation(num_samples)
                        class_data    = class_data[perm]
                        class_targets = class_targets[perm]

                        start = 0
                        for c in range(num_clients):
                            end = min(start + counts[c], num_samples)

                            client_data[c].append(class_data[start:end])
                            client_targets[c].append(class_targets[start:end])

                            start = end

                    # ------------------------------
                    # CONCAT
                    # ------------------------------
                    for c in range(num_clients):
                        client_data[c] = (
                            np.concatenate(client_data[c])
                            if client_data[c]
                            else np.empty((0,), dtype=np.float32)
                        )
                        client_targets[c] = (
                            np.concatenate(client_targets[c])
                            if client_targets[c]
                            else np.empty((0,), dtype=np.int64)
                        )

                    # ------------------------------
                    # TRAIN → equal samples
                    # ------------------------------
                    if split == "train":
                        total_task_samples = sum(len(client_data[c]) for c in range(num_clients))
                        target_per_client  = total_task_samples // num_clients

                        all_data_pool    = np.concatenate(client_data)
                        all_targets_pool = np.concatenate(client_targets)

                        for c in range(num_clients):
                            n = len(client_data[c])

                            if n == 0:
                                idx = np.random.choice(len(all_data_pool), size=target_per_client, replace=True)
                            elif n >= target_per_client:
                                idx = np.random.choice(n, size=target_per_client, replace=False)
                            else:
                                idx = np.random.choice(n, size=target_per_client, replace=True)

                            task_data[c]    = client_data[c][idx] if n > 0 else all_data_pool[idx]
                            task_targets[c] = client_targets[c][idx] if n > 0 else all_targets_pool[idx]

                    else:
                        total_task_samples = sum(len(client_data[c]) for c in range(num_clients))
                        target_per_client  = total_task_samples // num_clients

                        all_data_pool    = np.concatenate(client_data)
                        all_targets_pool = np.concatenate(client_targets)

                        for c in range(num_clients):
                            n = len(client_data[c])

                            if n == 0:
                                # 🔥 SAME FIX AS TRAIN
                                idx = np.random.choice(len(all_data_pool), size=target_per_client, replace=True)
                                task_data[c]    = all_data_pool[idx]
                                task_targets[c] = all_targets_pool[idx]

                            elif n >= target_per_client:
                                idx = np.random.choice(n, size=target_per_client, replace=False)
                                task_data[c]    = client_data[c][idx]
                                task_targets[c] = client_targets[c][idx]

                            else:
                                idx = np.random.choice(n, size=target_per_client, replace=True)
                                task_data[c]    = client_data[c][idx]
                                task_targets[c] = client_targets[c][idx]

                    min_samples = min(len(td) for td in task_data)

                # ------------------------------
                # SAVE
                # ------------------------------
                if format == "pytorch":
                    getattr(self, f"{split}_data").append(
                        [torch.tensor(td) for td in task_data]
                    )
                    getattr(self, f"{split}_targets").append(
                        [torch.tensor(tt) for tt in task_targets]
                    )
                else:
                    getattr(self, f"{split}_data").append(task_data)
                    getattr(self, f"{split}_targets").append(task_targets)

                # ------------------------------
                # DEBUG PRINT
                # ------------------------------
                print(f"\n[{split.upper()}] Task {task} summary:")
                total_samples = sum(len(td) for td in task_data)
                print(f"Total samples: {total_samples}")
                print(f"{'Client':<10} {'Samples':>8}  Class Distribution (count | proportion)")
                print("-"*80)

                for c in range(num_clients):
                    labels = task_targets[c]
                    total  = len(labels)

                    if total == 0:
                        print(f"Client {c:<6} {'0':>8} EMPTY")
                        continue

                    unique, counts = np.unique(labels, return_counts=True)

                    dist_str = []
                    for u, cnt in zip(unique, counts):
                        prop = cnt / total
                        dist_str.append(f"cls{int(u)}:{cnt} ({prop:.2f})")

                    print(f"Client {c:<6} {total:>8}  " + "  ".join(dist_str))

                print("-"*80)
                print(f"Per-client samples: {len(task_data[0])}")

        print("\n==================== DATA SPLIT DONE ====================")

    def get_cur_dataloaders_oos(self, task: int):
        self.cur_train_loaders, self.cur_test_loaders = [], []
        for split in ["train", "test"]:
            for client_idx in range(self.num_clients):
                cur_dataset = deepcopy(getattr(self, f"{split}_dataset"))

                tasks_data       = getattr(self, f"{split}_data")[task]
                cur_input_ids    = tasks_data["input_ids"][client_idx]
                cur_input_attmat = tasks_data["attention_mask"][client_idx]

                cur_data = {"input_ids": cur_input_ids, "attention_mask": cur_input_attmat}

                cur_dataset.data    = BatchEncoding(cur_data)
                cur_dataset.targets = getattr(self, f"{split}_targets")[task][client_idx]

                getattr(self, f"cur_{split}_loaders").append(
                    DataLoader(cur_dataset, self.batch_size, shuffle=True)
                )

        return self.cur_train_loaders, self.cur_test_loaders

    def get_cur_dataloaders(self, task: int):
        self.cur_train_loaders, self.cur_test_loaders = [], []
        for split in ["train", "test"]:
            for client_idx in range(self.num_clients):
                cur_dataset = deepcopy(getattr(self, f"{split}_dataset"))

                cur_dataset.data    = getattr(self, f"{split}_data")[task][client_idx]
                cur_dataset.targets = getattr(self, f"{split}_targets")[task][client_idx]

                getattr(self, f"cur_{split}_loaders").append(
                    DataLoader(cur_dataset, self.batch_size, shuffle=True)
                )

        return self.cur_train_loaders, self.cur_test_loaders