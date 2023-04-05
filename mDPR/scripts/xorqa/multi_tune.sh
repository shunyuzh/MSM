## Train: multi-lingual fine-tune for XOR

INIT_MODEL=${1:-'output/msm/best.pt'}
OUTPUT_DIR=${2:-'output/msm'}
MODEL_CFG=${3:-'model/msm'}
POOLER=${4:-'avg'}

outdir=$OUTPUT_DIR
model_cfg=$MODEL_CFG
init_model=${INIT_MODEL}

python -m torch.distributed.launch --nproc_per_node=8 train_dense_encoder.py \
--max_grad_norm 2.0 --encoder_model_type hf_bert --pretrained_model_cfg ${model_cfg} \
--seed 42 --sequence_length 256 --warmup_steps 1237 --batch_size 16 --dev_batch_size 64 \
--train_file data/xorqa/xorqa_dpr_data_query=L_hard_negative=1/dpr_train_data.json \
--dev_file data/xorqa/xorqa_dpr_data_query=L_hard_negative=1/dpr_dev_data.json \
--output_dir ${outdir} --learning_rate 2e-05 --num_train_epochs 40 --val_av_rank_start_epoch 30 \
--fp16 --pooling $POOLER --model_file ${init_model} --restart

# ----------------xorqa--------------------------------------------------------------
for i in {0..7}; do
  echo "xorqa inference on GPU $i"
	CUDA_VISIBLE_DEVICES=${i} python generate_dense_embeddings.py --model_file ${outdir}/best.pt --shard_id ${i} --num_shards 8 --fp16  \
	--ctx_file data/xorqa/en_wiki.tsv --out_file ${outdir}/emb_xorqa/emb --log_file ${outdir}/emb_xorqa/logger_${i}.log --batch_size 1024 &
	echo "xorqa inference finished on GPU $i"
done
wait

echo "xorqa retrieve on GPU 0"
CUDA_VISIBLE_DEVICES=0 python dense_retriever.py --data xorqa --n-docs 100 --fp16 --model_file ${outdir}/best.pt --batch_size 1024 \
--encoded_ctx_file ${outdir}/emb_xorqa/emb_\*  --ctx_file data/xorqa/en_wiki.tsv --log_file ${outdir}/results/logger.log \
--input_file data/xorqa/dev.jsonl data/xorqa/test.jsonl --out_file ${outdir}/results/dev.json ${outdir}/results/test.json
echo "xorqa retrieve finished on GPU 0"
