# python transfer.py \
# --dataset_to CIFAR10Distorted \
# --network Unet \
# --model_from models/CIFAR10_resnet34.ckpt \
# --cuda \
# --save_best \
# --lr 0.1 \
# --f_stats 0.1 \
# --size 2048 \
# --batch_size 256 \
# --num_epochs 10 \
# --reset

for lr in 0.01 0.05 0.1
    do
    # for f in 0.01 0.05 0.1 0.2
    for f in 1e-7 5e-7 1e-6 5e-6 1e-5 5e-5 1e-4 
        do
        python transfer.py \
        --dataset_to CIFAR10Distorted \
        --network Unet \
        --model_from models/CIFAR10_resnet34.ckpt \
        --cuda \
        --save_best \
        --lr $lr \
        --f_stats $f \
        --size 2048 \
        --batch_size 256 \
        --num_epochs 50 \
        --reset
    done
done

# --size 4096 \
# --unsupervised

# --dataset PBCBarcelona \
# --model_ckpt models/PBCBarcelona_resnet34_no_augmentation \
# --batch_size 4 \
