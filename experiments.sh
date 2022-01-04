run='python -m IPython --no-banner --no-confirm-exit'

for i in {1..50}
do
    $run experiments.py -- \
    --json experiments/unsupervised_reg.json
done
