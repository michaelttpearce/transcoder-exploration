from huggingface_hub import snapshot_download

# Dufensky, Chlenski, Nanda (2024) gpt-2 transcoders
repo_id = "pchlenski/gpt2-transcoders"
local_dir = "dufensky_transcoders"
snapshot_download(repo_id, local_dir=local_dir)