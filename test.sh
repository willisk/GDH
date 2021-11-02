run='python -m IPython --no-banner --no-confirm-exit'
$run transfer.py -- \
--dataset_to PBCBarcelona_4x \
--network Unet \
--model_from models/Cytomorphology_4x_resnet34.ckpt \
--cuda \
--save_best \
--lr 0.01 \
--f_stats 1e-7 \
--size 512 \
--batch_size 64 \
--num_epochs 300 \
--reset

# python invert.py \
# --model models/SVHN_resnet34.ckpt \
# --cuda \
# --save_best \
# --lr 0.1 \
# --f_stats 0 \
# --batch_size 512 \
# --num_epochs 2000 \
# --reset