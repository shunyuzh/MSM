# pipeline to evaluate tatoeba

MODEL_DIR=${1:-"model/msm"}
EXP=${2:-"msm"}

#------------------------ lareqa -------------------------
# 1. infer and test the model
echo "EXP: checkpoints_convert/${EXP}"
bash scripts/train_lareqa.sh $MODEL_DIR xlm-roberta-base ${EXP} 1 &
wait

# 2. print and output the results
INFO="LR2e-5_EPOCH3.0_LEN352"
bash scripts/run_eval_lareqa.sh outputs/lareqa/${EXP}_${INFO} xlm-roberta-base 1 checkpoint-best &
wait
echo "Finished evaluations. "