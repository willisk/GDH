run='python -m IPython --no-banner --no-confirm-exit'

# $run transfer.py -- \
# --dataset_to Cytomorphology-4x \
# --network Unet \
# --network BaselineColorMatrix \
# --model_from models/Cytomorphology-PBC-resnet34.ckpt \
# --cuda \
# --lr 1e-3 \
# --f_stats 1e-7 \
# --batch_size 64 \
# --num_epochs 3 \
# --reset

$run transfer.py -- \
--dataset_to PBCBarcelona-4x \
--network Unet \
--model_from models/Cytomorphology-4x_Resnet34.ckpt \
--size 128 \
--lr 1e-1 \
--f_stats 0 \
--batch_size 64 \
--num_epochs 100 \
--retrain_baseline \
--reset

# $run train.py -- \
# --network Resnet34 \
# --dataset Cytomorphology-4x \
# --lr 0.01 \
# --batch_size 64 \
# --num_epochs 5 \

# $run train.py -- \
# --network Resnet34 \
# --dataset Cytomorphology-4x-PBC \
# --lr 0.01 \
# --batch_size 64 \
# --num_epochs 5 \

# for lr in 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7
#     do
#     for f_st in 0 1e-7
#         do

#         $run transfer.py -- \
#         --dataset_to Cytomorphology_PBC \
#         --network Unet \
#         --network BaselineColorMatrix \
#         --model_from models/Cytomorphology_4x_resnet34.ckpt \
#         --cuda \
#         --lr $lr \
#         --f_stats $f_st \
#         --batch_size 64 \
#         --num_epochs 30

#         # --dataset_to CIFAR10Distorted \
#         # --model_from models/CIFAR10_resnet34.ckpt \
#     done
# done
