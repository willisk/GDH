# export CUDA_VISIBLE_DEVICES=0

run='python -m IPython --no-banner --no-confirm-exit'

$run prediction.py -- \
--network Resnet34 \
--dataset Cytomorphology-4x \
--lr 0.01 \
--batch_size 64 \
--num_epochs 20 \
--save_best \
--predic 0 \
--dataset_from Cytomorphology-4x \
--dataset_to PBCBarcelona-4x \
--model_from models/Cytomorphology-4x_Resnet18.ckpt \
--model_to transfer/Unet_Cytomorphology-4x_Resnet18_to_PBCBarcelona-4x_lr=1e-03_f-stats=1e-07/model.ckpt \
