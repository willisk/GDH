run='python -m IPython --no-banner --no-confirm-exit'

for size in 128 512 2048 8192 32768
    do

    $run transfer.py -- \
    --dataset_to PBCBarcelona-4x \
    --network Unet \
    --model_from models/Cytomorphology-4x_Resnet34.ckpt \
    --cuda \
    --lr 1e-3 \
    --f_stats 1e-7 \
    --size $size \
    --batch_size 64 \
    --num_epochs 30 \

    # --reset \

done


for lr in 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7
    do
    for f_stats in 0 1e-7
        do

        $run transfer.py -- \
        --dataset_to PBCBarcelona-4x \
        --network Unet \
        --model_from models/Cytomorphology_4x_resnet34.ckpt \
        --cuda \
        --lr $lr \
        --f_stats $f_stats \
        --batch_size 64 \
        --num_epochs 30

    done
done