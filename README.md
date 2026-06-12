# fed-mammoth2 - A Framework for Federated Continual Learning

## Setup

- Use `main.py` to run experiments.
- The mandatory arguments are:
  - `--model`
  - `--dataset`
  - `--network`
- New datasets can be added to the `_datasets/` folder.
- New models can be added to the `_models/` folder.
- New networks can be added to the `_networks/` folder.
- WandB logging can be enabled using `--wandb True`.

---

## Available Datasets

### Vision Datasets

- `seq-cifar10`
- `seq-cifar100`
- `seq-tinyimagenet`
- `seq-imagenetr`
- `seq-imageneta`
- `seq-cub200`
- `seq-cars`
- `seq-eurosat`
- `seq-isic`
- `seq-mnist`

### Text Datasets

- `seq-oos`

---

## Available Models

- `fedavg`
- `regmean`
- `ccvr`
- `der`
- `derpp`
- `ewc`
- `lwf`
- `fedproto`
- `target`
- `lorm`

---

## Available Networks

Examples:

- `resnet18_taskil`
- `vit`

Additional architectures can be found in the `_networks/` directory.

---

# Running Experiments

The framework uses the following command structure:

```bash
python3 -u main.py \
    --dataset seq-cifar100 \
    --network resnet18_taskil \
    --batch_size 32 \
    --lr 0.005 \
    --num_clients 5 \
    --num_comm_rounds 2 \
    --num_epochs 1 \
    --wandb False \
    --distribution_alpha 100000 \
    --model regmean
```

---

## Parameter Description

| Argument | Description |
|-----------|-------------|
| `--dataset` | Continual learning dataset |
| `--network` | Backbone network architecture |
| `--batch_size` | Local training batch size |
| `--lr` | Learning rate |
| `--num_clients` | Number of federated clients |
| `--num_comm_rounds` | Number of communication rounds |
| `--num_epochs` | Local training epochs per communication round |
| `--distribution_alpha` | Dirichlet parameter controlling client data heterogeneity |
| `--model` | Federated continual learning algorithm |
| `--wandb` | Enable or disable WandB logging |

---

## Example Runs

### FedAvg

```bash
python3 -u main.py \
    --dataset seq-cifar100 \
    --network resnet18_taskil \
    --batch_size 32 \
    --lr 0.005 \
    --num_clients 5 \
    --num_comm_rounds 2 \
    --num_epochs 1 \
    --wandb False \
    --distribution_alpha 100000 \
    --model fedavg
```

### RegMean

```bash
python3 -u main.py \
    --dataset seq-cifar100 \
    --network resnet18_taskil \
    --batch_size 32 \
    --lr 0.005 \
    --num_clients 5 \
    --num_comm_rounds 2 \
    --num_epochs 1 \
    --wandb False \
    --distribution_alpha 100000 \
    --model regmean
```

### CCVR

```bash
python3 -u main.py \
    --dataset seq-cifar100 \
    --network resnet18_taskil \
    --batch_size 32 \
    --lr 0.005 \
    --num_clients 5 \
    --num_comm_rounds 2 \
    --num_epochs 1 \
    --wandb False \
    --distribution_alpha 100000 \
    --model ccvr
```

### DER

```bash
python3 -u main.py \
    --dataset seq-cifar100 \
    --network resnet18_taskil \
    --batch_size 32 \
    --lr 0.005 \
    --num_clients 5 \
    --num_comm_rounds 2 \
    --num_epochs 1 \
    --wandb False \
    --distribution_alpha 100000 \
    --model der
```

### DER++

```bash
python3 -u main.py \
    --dataset seq-cifar100 \
    --network resnet18_taskil \
    --batch_size 32 \
    --lr 0.005 \
    --num_clients 5 \
    --num_comm_rounds 2 \
    --num_epochs 1 \
    --wandb False \
    --distribution_alpha 100000 \
    --model derpp
```

### EWC

```bash
python3 -u main.py \
    --dataset seq-cifar100 \
    --network resnet18_taskil \
    --batch_size 32 \
    --lr 0.005 \
    --num_clients 5 \
    --num_comm_rounds 2 \
    --num_epochs 1 \
    --wandb False \
    --distribution_alpha 100000 \
    --model ewc
```

### LwF

```bash
python3 -u main.py \
    --dataset seq-cifar100 \
    --network resnet18_taskil \
    --batch_size 32 \
    --lr 0.005 \
    --num_clients 5 \
    --num_comm_rounds 2 \
    --num_epochs 1 \
    --wandb False \
    --distribution_alpha 100000 \
    --model lwf
```

---

## Directory Structure

```text
fed-mammoth2/
├── _datasets/
├── _models/
├── _networks/
├── main.py
├── utils/
└── README.md
```

---

## Adding New Components

### New Dataset

Add the dataset implementation to:

```text
_datasets/
```

and register it in the dataset registry.

### New Model

Add the model implementation to:

```text
_models/
```

and register it in the model registry.

### New Network

Add the network implementation to:

```text
_networks/
```

and register it in the network registry.

---

## Notes

- `distribution_alpha = 100000` approximates an IID client distribution.
- Smaller values of `distribution_alpha` create increasingly non-IID client distributions.
- All experiments are launched through `main.py`.
- WandB logging is optional and can be disabled using:

```bash
--wandb False
```
