# Makefile - 支持跳过步骤 & 虚拟环境
.PHONY: data features model evaluate all sweep clean help

# ========================
# 🔧 配置区
# ========================

# 定义 Python 解释器路径（根据操作系统调整）
# Windows:
PYTHON := ../.venv/Scripts/python.exe
# Linux/macOS（取消注释）:
# PYTHON := .venv/bin/python

# 默认参数
N_ESTIMATORS ?= 100
MAX_DEPTH ?= 5

# 控制是否跳过某些步骤（默认不跳过）
SKIP_DATA ?= false
SKIP_FEATURES ?= false
SKIP_MODEL ?= false


# ========================
# 📦 数据与特征
# ========================

# Step 1: 获取数据
data:
	@echo "🔍 Step 1: 获取原始数据"
	@if [ "$(SKIP_DATA)" = "true" ]; then \
		echo "⏭️  SKIP_DATA=true，跳过数据获取步骤"; \
	else \
		"$(PYTHON)" src/data/make_dataset.py; \
	fi

# Step 2: 特征工程
features: data
	@echo "🔧 Step 2: 构建特征"
	@if [ "$(SKIP_FEATURES)" = "true" ]; then \
		echo "⏭️  SKIP_FEATURES=true，跳过特征工程步骤"; \
	else \
		"$(PYTHON)" src/features/build_features.py; \
	fi


# ========================
# 🤖 模型训练与评估
# ========================

# Step 3: 训练模型
model: features
	@echo "🤖 Step 3-1: 训练模型 n_estimators=$(N_ESTIMATORS), max_depth=$(MAX_DEPTH)"
	@if [ "$(SKIP_MODEL)" = "true" ]; then \
		echo "⏭️  SKIP_FEATURES=true，跳过特征工程步骤"; \
	else \
		"$(PYTHON)" src/models/train_model.py --n_estimators $(N_ESTIMATORS) --max_depth $(MAX_DEPTH); \
	fi

# Step 4: 模型评估
evaluate: model
	@echo "📈 Step 4: 评估模型 n_estimators=$(N_ESTIMATORS), max_depth=$(MAX_DEPTH)"
	"$(PYTHON)" src/evaluate/evaluate.py --n_estimators $(N_ESTIMATORS) --max_depth $(MAX_DEPTH)


# Step 5: 多参数扫描训练+评估
sweep: features
	@echo "🧠 Step 3-2: 开始参数扫描"
	bash src/scripts/sweep.sh "$(PYTHON)"


# ========================
# 🚀 全流程 & 工具
# ========================

# 全流程（默认参数）
all: data features model evaluate
	@echo "🎉 训练流水线执行完成！"


# 清理数据
clean:
	@echo "🧹 正在清理生成的数据..."
	rm -rf data/processed/*
	rm -rf models/*
	rm -rf reports/*
	@echo "✅ 清理完成"


# 查看帮助
help:
	@echo "可用命令："
	@echo ""
	@echo "  make data"
	@echo "      - 获取数据"
	@echo "      - 可加 SKIP_DATA=true 跳过（若数据已存在）"
	@echo ""
	@echo "  make features"
	@echo "      - 特征工程"
	@echo "      - 可加 SKIP_DATA=true 跳过 data 步骤"
	@echo ""
	@echo "  make model"
	@echo "      - 训练模型（使用默认参数）"
	@echo "      - 示例：make model N_ESTIMATORS=150 MAX_DEPTH=8 SKIP_DATA=true SKIP_FEATURES=true"
	@echo ""
	@echo "  make evaluate"
	@echo "      - 评估模型"
	@echo "      - 支持参数和 SKIP_DATA=true SKIP_FEATURES=true SKIP_MODEL=true"
	@echo ""
	@echo "  make sweep"
	@echo "      - 多参数扫描训练"
	@echo "      - 支持 SKIP_DATA=true SKIP_FEATURES=true"
	@echo ""
	@echo "  make all"
	@echo "      - 完整流水线"
	@echo "      - 支持 SKIP_DATA 和 SKIP_FEATURES"
	@echo ""
	@echo "  make clean"
	@echo "      - 清理生成文件"
	@echo ""
	@echo "  make help"
	@echo "      - 查看帮助"
	@echo ""
	@echo "💡 提示：确保已安装依赖：pip install -r requirements.txt"
	@echo "💡 当前使用 Python: $(PYTHON)"