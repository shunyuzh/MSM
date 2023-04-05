OUTPUT_DIR=${1:-'output/msm'}

echo "Started $OUTPUT_DIR"
python eval/mrtydi.py --pred_file ${OUTPUT_DIR}/results
python eval/xorqa.py --pred_file ${OUTPUT_DIR}/results/dev.json


