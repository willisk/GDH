run='python -m IPython --no-banner --no-confirm-exit'

for i in {1..10}
do
$run experiments.py -- \
--json experiments/baseline_sizes.json 

# --reload_results
done
