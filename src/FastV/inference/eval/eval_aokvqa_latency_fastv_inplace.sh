export CUDA_VISIBLE_DEVICES=1
# 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
model_path=liuhaotian/llava-v1.5-7b
output_path=different
mkdir -p $output_path

# rank_list=72 # rank equals to (1-R)*N_Image_Tokens, R=(87.5% 75% 50% 25%)
# Ks=2

for rank in 72; do
    for k in 2 4; do
        for h in 2 6 10 27; do
        for recent in 0.6; do
           for heavy in 0.5; do
            # auto download the ocrvqa dataset
            python ./src/FastV/inference/eval/inference_aokvqa.py \
                --model-path $model_path \
                --use-fast-v \
                --fast-v-inplace \
                --fast-v-sys-length 36 \
                --fast-v-image-token-length 576 \
                --fast-v-attention-rank $rank \
                --fast-v-agg-layer $k \
                --output-path $output_path/aokvqa_7b_FASTV_4bit_inplace_${rank}_${k}_${recent}_${heavy}_${h}.json  \
                --h2-user-prompt \
                --h2-system-prompt \
                --h2-agg-layer $h \
                --recent-budget-ratio $recent\
                --heavy-budget-ratio $heavy\
                # --same-layer
                # --h2-user-system-prompt
            done
            done
        done
    done
done


#Baseline
# python ./src/FastV/inference/eval/inference_aokvqa.py \
#     --model-path $model_path \
#     --output-path $output_path/aokvqa_7b_baseline.json 
