import os
import time
from huggingface_hub import snapshot_download

# 强行指定国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 彻底封杀那个动不动就崩溃的所谓“高速引擎”
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" 

repo_id = "huiwon/calvin_task_ABC_D"
local_dir = "data/calvin_raw/calvin_task_ABC_D"

print("🚀 开启【终极求稳】下载模式...")
print("（已关闭并发假死机制，拒绝快捷方式，只要实体文件！）")

while True:
    try:
        # 这个函数只要能完整跑完不报错，它内部就会把仓库的哈希值和本地文件逐一核对
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            resume_download=True,
            local_dir_use_symlinks=False,  # 核心改动：拒绝快捷方式，强制下载实体文件！
            max_workers=2                  # 核心改动：降低并发，宁愿慢一点，绝不让线程静默猝死
        )
        print("\n🎉 这次是真·下完了！哈希校验全部通过，没有骗人！")
        break
    except Exception as e:
        print(f"\n⚠️ 狗网络又断了: {e}")
        print("🔄 5秒后自动原地复活续传...")
        time.sleep(5)
