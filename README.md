# FastV + HeavyHitters Inference on A-OKVQA

This project evaluates the effectiveness of integrating **FastV** and **Heavy-Hitter** token pruning strategies for efficient inference on the A-OKVQA dataset using a modified LLaVA model.

---

## 🔧 Run the Inference Script

Use the following command to run inference with your custom configuration. Each argument is explained in-line:

```bash
python ./src/FastV/inference/eval/inference_aokvqa.py \
    --model-path $model_path \                     # Path to the pretrained model checkpoint
    --use-fast-v \                                 # Enable FastV token pruning
    --fast-v-inplace \                             # Apply token pruning in-place
    --fast-v-sys-length 36 \                       # Reserved length for system prompt tokens (default: 36)
    --fast-v-image-token-length 576 \              # Number of image tokens passed to the model (e.g., 24x24 = 576)
    --fast-v-attention-rank $rank \                # Top R image tokens to retain (R = attention rank)
    --fast-v-agg-layer $k \                        # Layer at which image token pruning begins
    --output-path $output_path/aokvqa_7b_FASTV_4bit_inplace_${rank}_${k}_${recent}_${heavy}_${h}.json \  # Output JSON path
    --h2-user-prompt \                             # Apply heavy-hitter only to user-prompt
    --h2-system-prompt \                           # Apply Heavy-Hitter to system prompt
    --h2-agg-layer $h \                            # Layer at which text token pruning begins
    --recent-budget-ratio $recent \                # Ratio of most recent tokens retained during generation (e.g., 0.5)
    --heavy-budget-ratio $heavy                    # Ratio of high-attention tokens retained (e.g., 0.6)
    # --same-layer                                 # (Optional) Prune text and image tokens at the same layer
