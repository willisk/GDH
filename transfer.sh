# for f1 in 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 0
#     do
#     for f2 in 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 0
#         do
#         python transfer.py \
#         --dataset_to CIFAR10Distorted \
#         --network Unet \
#         --model_from models/CIFAR10_resnet34.ckpt \
#         --cuda \
#         --save_best \
#         --lr 0.01 \
#         --f_stats 1e-6 \
#         --f_norm $f1 \
#         --f_tv $f2 \
#         --size 2048 \
#         --batch_size 256 \
#         --num_epochs 100 \
#         --reset
#     done
# done

for lr in 0.1
    do
    # for f in 0 1e-7 3e-7 7e-7 1e-6 3e-6
    for f_st in 0 1e-7 1e-6 1e-5 1e-4
        do
        python transfer.py \
        --dataset_to MNIST \
        --network Unet \
        --model_from models/SVHN_resnet34.ckpt \
        --cuda \
        --lr $lr \
        --f_stats $f_st \
        --size 2048 \
        --batch_size 256 \
        --num_epochs 500 \
        --reset

        # --resume_training

        # --dataset_to CIFAR10Distorted \
        # --model_from models/CIFAR10_resnet34.ckpt \
    done
done

# --size 4096 \
# --unsupervised

# --dataset PBCBarcelona \
# --model_ckpt models/PBCBarcelona_resnet34_no_augmentation \
# --batch_size 4 \
