# python transfer.py \
# --dataset_to MNIST \
# --network Unet \
# --model_from models/SVHN_resnet34.ckpt \
# --cuda \
# --save_best \
# --lr 0.01 \
# --f_stats 1e-7 \
# --size 512 \
# --batch_size 128 \
# --num_epochs 20 \
# --reset

python invert.py \
--model models/SVHN_resnet34.ckpt \
--cuda \
--save_best \
--lr 0.1 \
--f_stats 0 \
--batch_size 512 \
--num_epochs 2000 \
--reset