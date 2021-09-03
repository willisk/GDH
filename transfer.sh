python transfer.py \
--dataset_to CIFAR10Distorted \
--network Unet \
--model_from models/CIFAR10_resnet34.ckpt \
--cuda \
--save_best \
--lr 0.01 \
--batch_size 64 \
--num_epochs 20 \
--reset \

# --unsupervised

# --dataset PBCBarcelona \
# --model_ckpt models/PBCBarcelona_resnet34_no_augmentation \
# --batch_size 4 \