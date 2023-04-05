
## XLM-R
outdir=runs/XLM-R
model_cfg=xlm-roberta-base

python -m torch.distributed.launch --nproc_per_node=8 train_dense_encoder.py \
--max_grad_norm 2.0 --encoder_model_type hf_bert --pretrained_model_cfg ${model_cfg} \
--seed 42 --sequence_length 256 --warmup_steps 1237 --batch_size 16 --dev_batch_size 64 \
--train_file data/nq/biencoder-nq-train.json --dev_file data/nq/biencoder-nq-dev.json \
--output_dir ${outdir} --learning_rate 2e-05 --num_train_epochs 40 --val_av_rank_start_epoch 30 \
--fp16 --pooling avg  #  /cls  /model-file

# ---------------------myrtdi-------------------------------
for lang in sw bn te th id ko fi ar ja ru en; do
	for i in {0..7}; do
		CUDA_VISIBLE_DEVICES=${i} python generate_dense_embeddings.py --model_file ${outdir}/best.pt --shard_id ${i} --num_shards 8 --fp16  \
		--ctx_file data/mrtydi/${lang}/collection/docs.jsonl --out_file ${outdir}/emb_mrtydi/${lang}/emb --batch_size 1024 &
	done
	wait

	CUDA_VISIBLE_DEVICES=0 python dense_retriever.py --data mrtydi --model_file ${outdir}/best.pt --n-docs 100 --fp16 \
	--encoded_ctx_file ${outdir}/emb_mrtydi/${lang}/emb_\* --log_file ${outdir}/results/logger.log --batch_size 1024 \
	--input_file data/mrtydi/${lang}/topic.test.tsv --out_file ${outdir}/results/test_${lang}.tsv
done


# ---------------------xorqa-------------------------------
for i in {0..7}; do
	CUDA_VISIBLE_DEVICES=${i} python generate_dense_embeddings.py --model_file ${outdir}/best.pt --shard_id ${i} --num_shards 8 --fp16  \
	--ctx_file data/xorqa/en_wiki.tsv --out_file ${outdir}/emb_xorqa/emb --log_file ${outdir}/emb_xorqa/logger_${i}.log --batch_size 1024 &
done
wait

CUDA_VISIBLE_DEVICES=0 python dense_retriever.py --data xorqa --n-docs 100 --fp16 --model_file ${outdir}/best.pt --batch_size 1024 \
--encoded_ctx_file ${outdir}/emb_xorqa/emb_\*  --ctx_file data/xorqa/en_wiki.tsv --log_file ${outdir}/results/logger.log \
--input_file data/xorqa/dev.jsonl data/xorqa/test.jsonl --out_file ${outdir}/results/dev.json ${outdir}/results/test.json




