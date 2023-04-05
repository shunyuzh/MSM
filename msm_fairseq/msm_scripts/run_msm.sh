# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Train on 8 * A100 80GB
LGS_NUM=108
TBS=2048                           # 2048
WARMUP_UPDATES=10000                # Warmup the learning rate over this many updates. BERT is 10000
NGPU=`nvidia-smi -L | wc -l`
INIT=true                       # false (true)

MONO_DIR=$1
DISK_DIR=$2
NAME=$3
LR=${4:-0.00004}
MAX_LENGTH=${5:-256}
DOC_LAYERS=${6:-2}
BS=${7:-128}
TOTAL_UPDATES=${8:-200000}   # Total number of training steps.

MODEL_SIZE=base           # could be (small, base, large)
if [ $MODEL_SIZE == "large" ]; then
  BS=$((BS/2))
fi

export OMP_NUM_THREADS=1
export MKL_THREADING_LAYER=GNU
set -e
ulimit -n 65535
PER_NODE_GPU=$NGPU
NODE_COUNT=1
NODE_INDEX=0
TOTAL_GPU=$((PER_NODE_GPU*NODE_COUNT))


echo "Starting pip install --editable . --user"
pip install --editable . --user
pip install tensorboardX
pip install transformers==4.7.0
echo "CURRENT_DIR ${PWD}"

# cc below
LGS="af,als,am,an,ar,arz,as,ast,az,bar,be,bg,bn,br,bs,ca,ceb,ckb,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,he,hi,hr,hu,hy,ia,id,is,it,ja,jv,ka,kk,km,kn,ko,ku,ky,la,lb,lo,lt,lv,mg,mk,ml,mn,mr,ms,my,nds,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,sa,scn,sco,sd,sh,si,sk,sl,so,sq,sr,su,sv,sw,ta,te,th,tl,tr,tt,ug,uk,ur,uz,vi,war,wuu,yi,zh"
LGS_NUM=108

EXP_NAME="${NAME}_SIZE${MODEL_SIZE}_LGS${LGS_NUM}_LENGTH${MAX_LENGTH}"
echo "EXP_NAME: $EXP_NAME"
MODEL_DIR="${DISK_DIR}/checkpoints_exp/${EXP_NAME}/"
mkdir -p $MODEL_DIR

UPDATE_FREQ=$(($TBS/$BS/$TOTAL_GPU))      # Increase the batch size 16x
PEAK_LR=$LR                          # Peak learning rate, adjust as needed
tokens_per_sample_for_sentence=$MAX_LENGTH
TOKENS_PER_SAMPLE=$tokens_per_sample_for_sentence  # Max sequence length
MAX_POSITIONS=512                    # Num. positional embeddings (usually same as above)
MAX_SENTENCES=$BS                    # Number of sequences per batch (batch size)

echo "Update_freq: $UPDATE_FREQ"
echo "Total batch size: $(($BS*$TOTAL_GPU*$UPDATE_FREQ))"

SUFFIX=""
if [ $NODE_INDEX == 0 ] && [ $INIT == "true" ] && [ ! -f $MODEL_DIR/checkpoint1.pt ]; then
    echo "copy xlmr to last"
    cp ${DISK_DIR}/models/xlmr_based/xlmr_${MODEL_SIZE}.pt $MODEL_DIR/checkpoint_last.pt
    SUFFIX="--reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer"
fi

if [ $NODE_COUNT != 1 ]; then
    SUFFIX="$SUFFIX --distributed-init-method tcp://${MASTER_HOST}:${MASTER_PORT} --distributed-world-size $((NODE_COUNT*PER_NODE_GPU)) --distributed-rank $((NODE_INDEX*PER_NODE_GPU)) --device-id 0"
fi

echo "SUFFIX $SUFFIX"

LOCAL_LOG_PATH=./log.txt
python train.py $MONO_DIR:$PARA_DIR --save-dir $MODEL_DIR \
    --fp16 --ddp-backend=no_c10d \
    --distributed-backend nccl \
    --task msm --criterion masked_lm_msm \
    --skip-invalid-size-inputs-valid-test \
    --arch msm_$MODEL_SIZE --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ --max-positions $MAX_POSITIONS \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 \
    --validate-interval 4 --save-interval 1 --num-workers 4 \
    --mlm-langs "$LGS" \
    --save-interval-updates 2000 \
    --disable-validation \
    --enable-l2-norm \
    --enable-projection-head \
    --enable-projection-head-bn \
    --enable-memory-bank \
    --contrastive-max-sentence $((BS*4)) \
    --enable-lg-specific-memory-bank \
    --cts-temp 0.1 \
    --all-gather-list-size 327680 \
    --tokens-per-sample-for-sentence $tokens_per_sample_for_sentence \
    --doc-data $MONO_DIR \
    --enable-docencoder \
    --enable-doc-head \
    --check-mode doc_block \
    --document-encoder-layers $DOC_LAYERS \
    --enable-doc-block \
    --doc-bs-multiply 4 \
    --tensorboard-logdir $AZUREML_TB_PATH \
    --batch-pure-lg-doc \
    --enable-balance \
    $SUFFIX 2>&1 | tee $LOCAL_LOG_PATH