
MODEL_DIR="models/$(date +'%s')"
PARAMS_PATH="training/params.yml"
N_CLASSES=2

export CUDA_VISIBLE_DEVICES=""


python -m training.experiment \
    --model-dir "$MODEL_DIR" \
    --params-path $PARAMS_PATH \
    "$@"