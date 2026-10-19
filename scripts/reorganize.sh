#!/bin/bash
# ============================================================
# ElysiaCV 项目结构重组脚本
# 
# 1. 合并 logs/outputs/work_dirs → outputs/
# 2. projects/cassava → projects/HERBS
# 3. 清理冗余文件
# 4. 更新所有路径引用
# ============================================================
set -e

ROOT="/data/hdd3/chuqi/ElysiaCV"
cd "$ROOT"

echo "=========================================="
echo "Phase 1: 目录重组"
echo "=========================================="

# 1a. 创建 outputs/checkpoints 并移动 work_dirs 内容
echo "[1a] 移动 work_dirs → outputs/checkpoints ..."
mkdir -p outputs/checkpoints
for d in work_dirs/*/; do
    name=$(basename "$d")
    if [ -d "$d" ] && [ "$name" != "." ]; then
        echo "  移动 $name ..."
        mv "$d" "outputs/checkpoints/$name"
    fi
done
rmdir work_dirs/ 2>/dev/null || echo "  work_dirs 非空或已删除"

# 1b. 合并 figures/heatmaps/confusion_matrices → outputs/figures/
echo "[1b] 合并图片目录 ..."
mkdir -p outputs/figures
# Move figures subdirectory content
if [ -d "outputs/figures" ] && [ "$(ls -A outputs/figures/ 2>/dev/null)" ]; then
    # Already has some files, flatten subdirs
    :
fi
# Move heatmaps 
if [ -d "outputs/heatmaps" ]; then
    for f in outputs/heatmaps/*.png; do
        if [ -f "$f" ]; then
            mv "$f" outputs/figures/
        fi
    done
    rmdir outputs/heatmaps/ 2>/dev/null || rm -rf outputs/heatmaps/
fi
# Move confusion_matrices
if [ -d "outputs/confusion_matrices" ]; then
    for f in outputs/confusion_matrices/*.png; do
        if [ -f "$f" ]; then
            mv "$f" outputs/figures/
        fi
    done
    rmdir outputs/confusion_matrices/ 2>/dev/null || rm -rf outputs/confusion_matrices/
fi

# 1c. 删除空的 logs/
echo "[1c] 删除 logs/ ..."
rm -rf logs/

# 1d. 清理 outputs/figures 中不再使用的文件
echo "[1d] 清理冗余图片 ..."
cd outputs/figures/
# 删除旧版 heatmap（已被 RISE 替换）
rm -f all_models_heatmaps.png all_models_heatmaps_readable.png 2>/dev/null
# 删除 debug 文件
rm -f backbone_energy_fixed.png debug_original.png 2>/dev/null
# 删除旧版 multistage readable 变体
rm -f herbs_swin_base_multistage_readable.png 2>/dev/null
cd "$ROOT"

echo ""
echo "=========================================="
echo "Phase 2: projects/cassava → projects/HERBS"
echo "=========================================="
echo "[2] 重命名 projects/cassava → projects/HERBS ..."
mv projects/cassava projects/HERBS

echo ""
echo "=========================================="
echo "Phase 3: 更新所有路径引用"
echo "=========================================="

# 3a. 更新配置文件中的 work_dirs → outputs/checkpoints
echo "[3a] 更新 configs/experiments/*.yaml ..."
find configs/experiments/ -name "*.yaml" -exec sed -i \
    's|output_dir: ./work_dirs/|output_dir: ./outputs/checkpoints/|g' {} \;

# 3b. 更新分类配置中的 work_dirs 引用
echo "[3b] 更新 configs/classification/*.yaml ..."
find configs/classification/ -name "*.yaml" -exec sed -i \
    's|work_dirs/|outputs/checkpoints/|g' {} \;

# 3c. 更新 tools/analyze.py 中的路径
echo "[3c] 更新 tools/analyze.py ..."
if [ -f tools/analyze.py ]; then
    sed -i 's|work_dirs/|outputs/checkpoints/|g' tools/analyze.py
    sed -i 's|outputs/heatmaps/|outputs/figures/|g' tools/analyze.py
    sed -i 's|outputs/confusion_matrices/|outputs/figures/|g' tools/analyze.py
fi

# 3d. 更新 scripts/ 中的路径
echo "[3d] 更新 scripts/*.py ..."
for f in scripts/visualize_heatmaps.py scripts/visualize_rise.py scripts/eval_trainings.py; do
    if [ -f "$f" ]; then
        sed -i 's|work_dirs/|outputs/checkpoints/|g' "$f"
        sed -i 's|outputs/heatmaps/|outputs/figures/|g' "$f"
        sed -i 's|outputs/confusion_matrices/|outputs/figures/|g' "$f"
    fi
done
if [ -f scripts/export_docx.py ]; then
    sed -i 's|work_dirs/|outputs/checkpoints/|g' scripts/export_docx.py
    sed -i 's|outputs/heatmaps/|outputs/figures/|g' scripts/export_docx.py
    sed -i 's|outputs/confusion_matrices/|outputs/figures/|g' scripts/export_docx.py
fi

# 3e. 更新脚本中的 projects/cassava → projects/HERBS
echo "[3e] 更新 projects 引用 ..."
find scripts/ -name "*.py" -exec sed -i 's|projects/cassava|projects/HERBS|g' {} \;

# 3f. 更新文档中的路径
echo "[3f] 更新 PAPER.md ..."
if [ -f PAPER.md ]; then
    sed -i 's|work_dirs/|outputs/checkpoints/|g' PAPER.md
    sed -i 's|outputs/heatmaps/|outputs/figures/|g' PAPER.md
    sed -i 's|outputs/confusion_matrices/|outputs/figures/|g' PAPER.md
    sed -i 's|outputs/figures/disease_samples.png|outputs/figures/disease_samples.png|g' PAPER.md
fi

echo "[3g] 更新 EXPERIMENT_REPORT.md ..."
if [ -f EXPERIMENT_REPORT.md ]; then
    sed -i 's|work_dirs/|outputs/checkpoints/|g' EXPERIMENT_REPORT.md
fi

echo "[3h] 更新 tools/README.md ..."
if [ -f tools/README.md ]; then
    sed -i 's|work_dirs/|outputs/checkpoints/|g' tools/README.md
fi

echo "[3i] 更新 projects/HERBS/AGENTS.md ..."
if [ -f projects/HERBS/AGENTS.md ]; then
    sed -i 's|projects/cassava|projects/HERBS|g' projects/HERBS/AGENTS.md
    sed -i 's|cassava|HERBS|g' projects/HERBS/AGENTS.md
    sed -i 's|work_dirs/|outputs/checkpoints/|g' projects/HERBS/AGENTS.md
fi

echo "[3j] 更新其他 docs/ ..."
find docs/ -name "*.md" -exec sed -i 's|work_dirs/|outputs/checkpoints/|g' {} \; 2>/dev/null
find docs/ -name "*.md" -exec sed -i 's|projects/cassava|projects/HERBS|g' {} \; 2>/dev/null

echo "[3k] 更新其他引用 ..."
find configs/ -name "*.md" -exec sed -i 's|work_dirs/|outputs/checkpoints/|g' {} \; 2>/dev/null
find configs/ -name "AGENTS.md" -exec sed -i 's|projects/cassava|projects/HERBS|g' {} \; 2>/dev/null

# 更新 AGENTS.md
if [ -f AGENTS.md ]; then
    sed -i 's|work_dirs/|outputs/checkpoints/|g' AGENTS.md
    sed -i 's|projects/cassava|projects/HERBS|g' AGENTS.md
fi

# 3l. 更新 .gitignore
echo "[3l] 更新 .gitignore ..."
sed -i 's|work_dirs/|outputs/checkpoints/|g' .gitignore 2>/dev/null
# 确保 *.pt 等仍然是 gitignored
if ! grep -q "*.pt" .gitignore 2>/dev/null; then
    echo "*.pt" >> .gitignore
fi
if ! grep -q "*.pth" .gitignore 2>/dev/null; then
    echo "*.pth" >> .gitignore
fi

echo ""
echo "=========================================="
echo "Phase 4: 验证与清理"
echo "=========================================="

# 确保不再有 work_dirs 引用残留
echo "[4a] 检查 work_dirs 残留引用 ..."
grep -rl "work_dirs" --include="*.py" --include="*.yaml" --include="*.sh" --include="*.md" \
    $(find . -maxdepth 1 -not -name ".git" -not -name "__pycache__" -type d | tr '\n' ' ') \
    2>/dev/null | grep -v ".pyc" | grep -v "reorganize" || echo "  ✓ 无残留引用"

echo ""
echo "[4b] 最终目录结构:"
echo "---"
echo "outputs/"
ls -la outputs/ 2>/dev/null | tail -n +4
echo ""
echo "outputs/figures/:"
ls -lh outputs/figures/ 2>/dev/null | tail -n +2
echo ""
echo "outputs/checkpoints/:"
ls outputs/checkpoints/ 2>/dev/null
echo ""
echo "projects/:"
ls projects/ 2>/dev/null
echo ""

echo "=========================================="
echo "重组完成！"
echo "=========================================="