# Post-Hoc Domain Adaptation via Guided Data Homogenization
implementation of
https://arxiv.org/abs/2104.03624


Install requirements
```bash
pip install -r requirements.txt
```

Train a model (CIFAR10)
```bash
python train.py \
--dataset CIFAR10 \
--network resnet34 \
--cuda \
--save_best \
--lr 0.01 \
--batch_size 64 \
--num_epochs 20
```

Transfer test for distorted CIFAR10
```bash
python transfer.py \
--dataset_to CIFAR10Distorted \
--network Unet \
--model_from models/CIFAR10_resnet34.ckpt \
--cuda \
--save_best \
--lr 0.01 \
--batch_size 64 \
--num_epochs 20
```