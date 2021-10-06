# export CUDA_VISIBLE_DEVICES=0
# --dataset PBCBarcelona_4x \

python train.py \
--dataset Cytomorphology_4x \
--network resnet34 \
--cuda \
--save_best \
--lr 0.01 \
--batch_size 64 \
--num_epochs 20 \
--reset \

# --resume_training

# --dataset PBCBarcelona \
# --model_ckpt models/PBCBarcelona_resnet34_no_augmentation \
# --batch_size 4 \