from huggingface_hub import hf_hub_download
import shutil

for split in ["train", "test", "val"]:
    file = f"{split}.csv"
    cache_path = hf_hub_download(repo_id="mks-logic/gender_prediction", repo_type="dataset", filename=file)
    shutil.copy(cache_path, f"./data/{file}")