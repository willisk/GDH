# export CUDA_VISIBLE_DEVICES=0

python train.py \
--dataset CIFAR10 \
--network resnet34 \
--cuda \
--save_best \
--lr 0.01 \
--batch_size 64 \
--num_epochs 4 \
--reset \

# --resume_training

# --dataset PBCBarcelona \
# --model_ckpt models/PBCBarcelona_resnet34_no_augmentation \
# --batch_size 4 \