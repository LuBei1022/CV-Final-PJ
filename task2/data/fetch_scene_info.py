"""只从远程 CALVIN zip 里抽取 scene_info.npy（几 KB），不下载整个 517GB。

原理：zip 的文件目录在文件末尾，remotezip 用 HTTP Range 请求只读目录、
再只取我们要的那个小文件。前提是服务器支持 Range 请求（freiburg 静态服务器通常支持）。

依赖：pip install remotezip

用法：
    python data/fetch_scene_info.py --out ./data/calvin_meta

成功后会得到 ./data/calvin_meta/scene_info.npy（训练集 = 环境 A/B/C 的区间）。
若失败（服务器不支持 Range / 网络受限），改用以下任一备选：
  1) 直接找老师要 scene_info.npy（他做这份数据集时一定有，最稳）。
  2) 在 AutoDL 上完整下载 task_ABC_D.zip 后取 training/scene_info.npy（很慢，517GB）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

ZIP_URL = "http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data/calvin_meta", help="保存目录（相对路径）")
    ap.add_argument("--url", default=ZIP_URL)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    try:
        from remotezip import RemoteZip
    except ImportError:
        raise SystemExit("请先安装: pip install remotezip")

    print(f"[fetch] 连接远程 zip（只读目录，不下载全部）: {args.url}")
    with RemoteZip(args.url) as z:
        # 训练集的 scene_info 才是 A/B/C；validation 的是 D
        candidates = [n for n in z.namelist()
                      if n.endswith("scene_info.npy") and "training" in n]
        if not candidates:
            candidates = [n for n in z.namelist() if n.endswith("scene_info.npy")]
        if not candidates:
            raise SystemExit("zip 里没找到 scene_info.npy，请改用备选方案。")
        target = candidates[0]
        print(f"[fetch] 抽取: {target}")
        z.extract(target, path=str(out))

    # 规整到 out/scene_info.npy
    extracted = next(out.rglob("scene_info.npy"))
    final = out / "scene_info.npy"
    if extracted != final:
        extracted.replace(final)
    print(f"[done] 已保存: {final}")


if __name__ == "__main__":
    main()
