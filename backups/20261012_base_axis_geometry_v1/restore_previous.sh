#!/usr/bin/env bash
set -euo pipefail
snapshot_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$snapshot_dir/../.." && pwd)"
if ps -eo comm=,args= | awk '$1 ~ /^(python|torchrun)/ && /train_(3dkdavis_affinity|base_axis_geometry_v1)[.]py/ {found=1} END {exit !found}'; then
  echo '请先停止 affinity/Base 训练，再恢复源码。' >&2
  exit 1
fi
(cd -- "$snapshot_dir" && sha256sum -c SHA256SUMS)
preserved_dir="$(mktemp -d "$snapshot_dir/before_restore.XXXXXX")"
for name in model.py comnet.py; do
  cp -p -- "$project_dir/$name" "$preserved_dir/"
  cp -p -- "$snapshot_dir/$name" "$project_dir/$name"
  cmp -- "$snapshot_dir/$name" "$project_dir/$name"
done
echo "已恢复本次修复前的 model.py 和 comnet.py；恢复前源码保留于 $preserved_dir"
echo '已有模型、数据、缓存和训练输出未改动。旧严格模型可继续使用原运行脚本。'
echo '新训练的 Base 权重应配合修复版源码使用，不能在恢复版源码中混用。'
