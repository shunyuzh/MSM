# Convert Fairseq checkpoint to Huggingface file
# Dependency: transformers==2.5.1

OUTPUT_NAME=$1
INPUT_NAME=$2
MODEL_BLOB=${3:-0}
SIZE=${4:-0}

RESOURCE=$MODEL_BLOB/models/xlmr${SIZE}_hfs
TARGET_FOLDER=$MODEL_BLOB/checkpoints_convert/$OUTPUT_NAME
INPUT_FOLDER=$MODEL_BLOB/checkpoints_ex/$INPUT_NAME

cd $MODEL_BLOB/msm_fairseq
echo  "Test env success"

mkdir -p $TARGET_FOLDER
convert(){
    name=$1
    echo "Started $name"
    OUTPUT_FOLDER=$TARGET_FOLDER/model$name
    mkdir -p $OUTPUT_FOLDER
    cp -r $RESOURCE/* $OUTPUT_FOLDER/
    cp $INPUT_FOLDER/*_${name}.pt $OUTPUT_FOLDER/model.pt
    python msm_utils/convert_roberta_original_pytorch_checkpoint_to_pytorch.py \
    --roberta_checkpoint_path $OUTPUT_FOLDER --pytorch_dump_folder_path $OUTPUT_FOLDER
    rm $OUTPUT_FOLDER/model.pt
    echo "Finished $name"
}

# Convert Fairseq checkpoint to Huggingface file
convert $NAME &
