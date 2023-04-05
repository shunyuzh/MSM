# pipeline to evaluate tatoeba

MODEL_DIR=${1:-"model/msm"}
EXP=${2:-"msm"}

#------------------------ mewslix -------------------------
# 1. infer and test the model
echo "EXP: checkpoints_convert/${EXP}"
bash scripts/train_mewslix.sh $MODEL_DIR xlm-roberta-base ${EXP} 1 &
wait

# 2. output the results
INFO="LR2e-5_EPOCH2_LEN64_BS16_ACC4"
bash scripts/run_eval_mewslix.sh outputs/mewslix/${EXP}_${INFO} xlm-roberta-base 1 &
wait
echo "Finished evaluations. "