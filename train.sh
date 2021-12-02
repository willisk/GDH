# export CUDA_VISIBLE_DEVICES=0
# --dataset Cytomorphology_PBC \

run='python -m IPython --no-banner --no-confirm-exit'
$run train.py -- \
--network resnet34 \
--dataset Cytomorphology_PBC \
--cuda \
--save_best \
--lr 0.01 \
--batch_size 64 \
--num_epochs 5 \

# --reset \

# --resume_training

# --dataset PBCBarcelona \
# --model_ckpt models/PBCBarcelona_resnet34_no_augmentation \
# --batch_size 4 \