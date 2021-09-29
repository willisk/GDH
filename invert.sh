
for f_reg in 0 1e-6
    do
    # for f_st in 0 1e-7 3e-7 7e-7 1e-6 3e-6 1e-5 1e-4
    for f_st in 7e-7 1e-6 3e-6 1e-5
        do
        python invert.py \
        --model models/SVHN_resnet34.ckpt \
        --cuda \
        --lr 0.1 \
        --f_reg $f_reg \
        --f_stats $f_st \
        --batch_size 256 \
        --num_epochs 2000 \

        # --reset
    done
done