export CUDA_VISIBLE_DEVICES=2
python ./src/FastV/inference/plot_inefficient_attention.py \
    --model-path "liuhaotian/llava-v1.5-7b" \
    --image-path "./src/LLaVA/images/llava_logo.png" \
    --prompt "Describe the image in details."\
    --output-path "./output_example_path"