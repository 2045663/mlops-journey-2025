#!/bin/bash
# sweep.sh - 参数扫描脚本（支持虚拟环境）
# 使用方法: ./sweep.sh [python_interpreter_path]
# 示例: ./sweep.sh ../.venv/Scripts/python.exe

# 获取传入的 Python 解释器路径，如果未传入则尝试自动检测或使用默认
PYTHON="${1:-python}"

# 检查是否能获取 Python 版本（验证解释器可用性）
if ! "$PYTHON" --version >/dev/null 2>&1; then
    echo "❌ 错误：无法执行 '$PYTHON'，请检查 Python 解释器路径是否正确"
    echo "💡 提示："
    echo "   Windows: .venv\\Scripts\\python.exe 或 .venv/Scripts/python.exe"
    echo "   Linux/macOS: .venv/bin/python"
    exit 1
fi

echo "🔍 使用 Python 解释器: $($PYTHON --version 2>&1)"
echo "🔍 开始参数扫描..."

# 超参数组合遍历
for n in 100 120 150; do
  for d in 5 7 9; do
    echo "🚀 训练：n_estimators=$n, max_depth=$d"
    "$PYTHON" src/models/train_model.py --n_estimators "$n" --max_depth "$d"

    if [ $? -ne 0 ]; then
      echo "❌ 训练失败：n_estimators=$n, max_depth=$d"
      exit 1
    fi

    echo "📊 评估：n_estimators=$n, max_depth=$d"
    "$PYTHON" src/evaluate/evaluate.py --n_estimators "$n" --max_depth "$d"

    if [ $? -ne 0 ]; then
      echo "❌ 评估失败：n_estimators=$n, max_depth=$d"
      exit 1
    fi

    echo "✅ 完成：n=$n, d=$d"
    echo "---"
  done
done

echo "🎉 参数扫描完成！共运行 9 次实验（训练 + 评估）"