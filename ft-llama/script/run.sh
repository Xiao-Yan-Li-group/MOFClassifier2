conda activate llama
llamafactory-cli train llama3_lora_sft.yaml
llamafactory-cli export merge_config.yaml
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
python LlamaFactory/scripts/vllm_infer.py \
  --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
  --dataset struc2str_all --save_name generated_predictions_struc2str_all.jsonl \
  --dataset_dir data --cutoff_len 4096 \
  --temperature 0.0 --top_p 1.0 --top_k 0 --max_new_tokens 16 \
  --repetition_penalty 1.0 --enable_thinking False --seed 120
