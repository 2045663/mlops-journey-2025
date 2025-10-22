# Makefile - 机器学习训练流水线
.PYTHON: data features model evaluate all

# Step 1: 获取数据
data:
	@echo "🔍 Step 1: 获取原始数据"
	python src/data/make_dataset.py

#  Step 2: 特征工程
features: data
	@echo "🔧 Step 2: 构建特征"
	python src/features/build_features.py

#  Step 3: 模型训练
model: features
	@echo "🤖 Step 3: 训练模型"
	python src/models/train_model.py

# Step 4: 模型评估
evaluate: model
	@echo "📈 Step 4: 评估模型"
	python src/evaluate/evaluate.py

# 全流程
all: data features model evaluate
	@echo "🎉 训练流水线执行完成！所有步骤均已成功运行。"

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
	@echo "  make data      - 获取数据"
	@echo "  make features  - 特征工程"
	@echo "  make model     - 训练模型"
	@echo "  make evaluate  - 评估模型"
	@echo "  make all       - 完整流水线"
	@echo "  make clean     - 清理生成文件"
	@echo "  make help      - 查看帮助"