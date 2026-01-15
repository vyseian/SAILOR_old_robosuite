from huggingface_hub import list_repo_files, hf_hub_download

repo_id = "arnavkj1995/SAILOR"
repo_type = "dataset"

# List all files in the repo
files = list_repo_files(repo_id=repo_id, repo_type=repo_type)

# Download only non-skipped files
for file in files:
    if file not in ["README.md", ".gitignore"]:
        print(f"Downloading {file}...")
        hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=file, local_dir=".", local_dir_use_symlinks=False)

# http://downloads.cs.stanford.edu/downloads/rt_benchmark/transport/mh/demo_v141.hdf5
# http://downloads.cs.stanford.edu/downloads/rt_benchmark/square/mh/demo_v141.hdf5