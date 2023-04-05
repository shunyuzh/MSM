# MSM

Code for MSM pre-training tailored for cross-lingual dense retrieval and the evaluation for the pre-trained models. Details can be found in our papers, [Modeling Sequential Sentence Relation to Improve Cross-lingual Dense Retrieval](https://arxiv.org/abs/2302.01626)



# Pre-training Stage

### Prepare the Environment

```
cd msm_fairseq
pip install --editable . --user
pip install tensorboardX
pip install transformers==4.7.0
```



### Data Pre-processing

Data should be preprocessed following the [language modeling format](https://github.com/facebookresearch/fairseq/blob/main/examples/language_model), i.e. each document should be separated by an empty line, as the following format.

```
Doc1's sentence1
Doc1's sentence2

Doc2's sentence1
Doc2's sentence2
...
```

Then the raw text data should be preprocessed with the XLM-R BPE model. 

```shell
DATA_PATH=/data/path
OUT_PATH=/out/path
MODEL_PATH=/bpe_model/path

all_lgs="ar en"
mkdir -p $DATA_PATH/processed_data
for SPLIT in $all_lgs; do
  echo $SPLIT
  python preprocess_data/spm_encode.py --model $MODEL_PATH/sentencepiece.bpe.model \
  --inputs $DATA_PATH/${SPLIT}.txt --outputs $OUT_PATH/bpe_data/train.${SPLIT} --ratio 1.0
done
```

Finally preprocess/binarize the data using the XLM-R dictionary.

```shell
INPUT_PATH=/input/path
OUTPUT_PATH=/output/path
MODEL_PATH=/model/path

all_lgs="ar en"
for lg in $all_lgs; do
  python3 preprocess.py \
    --task msm \
    --srcdict $MODEL_PATH/dict.txt \
    --only-source \
    --trainpref $INPUT_PATH/bpe \
    --destdir $OUTPUT_PATH \
    --workers 16 \
    --source-lang $lg
  mv "$OUTPUT_PATH/train.$lg-None.$lg.bin" "$OUTPUT_PATH/train.0.$lg.bin"
  mv "$OUTPUT_PATH/train.$lg-None.$lg.idx" "$OUTPUT_PATH/train.0.$lg.idx"
done
```



### Pre-training

The following bash lauch pre-training on 8 NVIDIA A100 80GB GPUs starting from XLM-R base. 

```
bash fairseq/run_msm.sh {DATA_DIR} {MODEL_DOR} {EXP_NAME} 
```

Then you can convert the saved checkpoints to the format as Huggingface models. 

```
bash fairseq/convert.sh {OUTPUT_NAME} {INPUT_NAME}
```



### Resource

You can download the pre-trained model from the following links.

[MSM-base model](https://unicoderrelease.blob.core.windows.net/denseretrieval/release/msmbase_hfs.tar.gz?sv=2020-10-02&st=2022-10-31T02%3A55%3A02Z&se=2030-11-01T02%3A55%3A00Z&sr=b&sp=r&sig=nikcLKAylP96LEAXbDnIsag0qmJ9T0kKiMyDseU4reU%3D)

[MSM-large model](https://unicoderrelease.blob.core.windows.net/denseretrieval/release/msmlarge_hfs.tar.gz?sv=2020-10-02&st=2022-10-31T02%3A53%3A15Z&se=2030-11-01T02%3A53%3A00Z&sr=b&sp=r&sig=Ya7CkF3%2BuFTqmKL80Oa6zOH7Nl4whIRJ9enHQJ9i0G8%3D)



# Fine-tuning and Evaluation

For Mr.TYDI and XOR Retrieve, we mainly follow [CCP](https://github.com/wuning0929/CCP_IJCAI22). And for LAReQA and Mewsli-X, we follow [XTREME](https://github.com/google-research/xtreme). We thank all the developers.



### Mr.TYDI

Prepare the environment and download the data:

```
cd mDPR
conda create -n mdpr python==3.7
pip install -r requirements.txt
bash download_data.sh
```

For Mr.TYDI, we use data from NQ or MS MARCO dataset for the fine-tuning of Cross-lingual zero-shot transfer. You can fine-tune a pre-trained multilingual model on the English data with the following command:

```
bash scripts/mrtydi/nq_tune.sh
bash scripts/mrtydi/marco_tune.sh
```

Then you can continue fine-tuning on Mr.TYDI train data for the model trained on MS MARCO, to get the results of Multi-lingual fine-tune setting. 

```
bash scripts/mrtydi/multi_tune.sh
```

Output the evaluation results:

```
bash eval/eval_scripts.sh
```



### XOR Retrieve

For Mr.TYDI, we use data from NQ dataset for the fine-tuning of Cross-lingual zero-shot transfer. You can fine-tune a pre-trained multilingual model on the English data with the following command:

```
bash scripts/xor/nq_tune.sh
```

Then you can continue fine-tuning on Mr.TYDI train data for the model trained on NQ, to get the results of Multi-lingual fine-tune setting. 

```
bash scripts/xor/multi_tune.sh
```

Output the evaluation results:

```
bash eval/eval_scripts.sh
```



### Mewsli-X

Prepare the environment and download the data, here you can download only the data related to our tasks.

```
cd xtreme
conda create -n xtreme python==3.7
bash install_tools.sh
bash scripts/download_data.sh
```

For Mewsli-X, it fine-tunes on a predefined set of English-only mention-entity pairs as in XTREME-R. You can fine-tune a pre-trained multilingual model on the data with the following command:

```
bash msm_scripts/mewslix.sh
```



### LAReQA

For LAReQA, it finetunes on the English QA pairs from SQuAD v1.1 train set. You can fine-tune a pre-trained multilingual model on the data with the following command:
```
bash msm_scripts/lareqa.sh
```



## References 

If you find our paper or the code helpful, please cite our paper.
```
@inproceedings{zhangmodeling,
  title={Modeling Sequential Sentence Relation to Improve Cross-lingual Dense Retrieval},
  author={Zhang, Shunyu and Liang, Yaobo and GONG, MING and Jiang, Daxin and Duan, Nan},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023}
}
```



## Acknowledgements

This code partly began with: [Fairseq](https://github.com/facebookresearch/fairseq), [CCP](https://github.com/wuning0929/CCP_IJCAI22), [Xtreme](https://github.com/google-research/xtreme)

We thank all the developers for doing most of the heavy-lifting.
