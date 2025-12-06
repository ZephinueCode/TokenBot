from huggingface_hub import snapshot_download
# from modelscope import snapshot_download
# snapshot_path = snapshot_download(repo_id="Qwen/Qwen3-VL-4B-Instruct",
#                                   local_dir="checkpoints/Qwen3-VL-4B-Instruct",
#                                   )

dataset_path = snapshot_download(
    repo_id="osunlp/Multimodal-Mind2Web",
    repo_type="dataset",
    local_dir="./data/mind2web",
)