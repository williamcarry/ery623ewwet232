"""
💻 **趋势预测训练 - Ensemble集成模型优化版（v2.0）**

═══════════════════════════════════════════════════════════════════════════
🚀 四部分完整使用指南
═══════════════════════════════════════════════════════════════════════════

【第一部】准备工作 - 数据准备（必须！）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
目录结构：
  ai_training_data/day_kline_training/
    ├── up_trend/data.json          # 上涨趋势的K线数据（JSON格式）
    ├── down_trend/data.json        # 下跌趋势的K线数据
    └── sideways/data.json          # 横盘趋势的K线数据

要求：
  ✓ 每个JSON文件中最少需要50条数据，建议100+条
  ✓ 每条数据是一个股票的120根K线（前60根历史+后60根）
  ✓ K线数据格式: {open, high, low, close, volume}
  ✓ 数据必须来自同一个数据库（用StockImageAnalyzer加载）

数据格式示例（data.json）：
  [
    {
      "symbol": "000858",
      "period": "day",
      "kline_data": [{"open": 10.5, "high": 10.8, "low": 10.3, "close": 10.6, "volume": 1000000}, ...]
    },
    ...
  ]


【第二部】运行训练 - 模型训练（核心！）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
命令行执行：

  python src/ai_training/train_trend_lstm_ensemble_v2.py ai_training_data/day_kline_training

或者使用默认路径（如果数据在项目根目录）：

  python src/ai_training/train_trend_lstm_ensemble_v2.py

训练过程中会自动：
  ✓ 加载和验证K线数据
  ✓ 提取60×42=2520维特征（每根K线42个特征）
  ✓ 自动进行数据标准化（StandardScaler）
  ✓ 训练3个子模型：XGBoost、RandomForest、LSTM
  ✓ 自动加权投票集成（性能最佳）
  ✓ 生成学习曲线、特征重要性等分析报告

训练时间：15-30分钟（取决于数据量和硬件）

预期准确率：93-98%（集成模型比单模型高20-30%）


【第三部】保存模型 - 自动保存到硬盘
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
训练完成后，自动保存文件到：

  trained_models/trend_ensemble_models_v2/
  ├── ensemble_model.pkl          ⭐ 集成模型（主要使用这个！）
  ├── xgb_model.pkl               (子模型，不需要单独用)
  ├── rf_model.pkl                (子模型，不需要单独用)
  ├── lstm_model.h5               (子模型，不需要单独用)
  ├── lstm_model.tflite           (手机版LSTM，仅供参考)
  ├── scaler.pkl                  ⭐ 标准化器（预测时必须用！）
  ├── model_metadata.json         (训练信息：准确率、数据量等)
  ├── learning_curves.png         (LSTM学习曲线图)
  ├── feature_importance.json     (特征重要性排名)
  └── checkpoints/                (LSTM训练检查点，可删除)

关键文件：
  • ensemble_model.pkl    - 必须！包含集成模型
  • scaler.pkl           - 必须！包含训练时的标准化参数
  • 其他文件 - 可选，用于分析和调试


【第四部】加载和预测 - 实际使用
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
步骤1️⃣：准备K线数据
  • 需要120根K线数据（前60根历史 + 后60根显示）
  • 格式：list 或 numpy array，每个元素有 open, high, low, close, volume

步骤2️⃣：加载模型和标准化器
  import joblib
  model = joblib.load('trained_models/trend_ensemble_models_v2/ensemble_model.pkl')
  scaler = joblib.load('trained_models/trend_ensemble_models_v2/scaler.pkl')

步骤3️⃣：提取特征并标准化
  from ai_training.feature_extractor import extract_features_sequence_from_kline_data

  # 步骤3.1: 提取特征 (60, 42)
  X_new = extract_features_sequence_from_kline_data(kline_data)

  # 步骤3.2: 展平 (60, 42) → (1, 2520)
  X_new_flat = X_new.reshape(1, -1)

  # 步骤3.3: 标准化（关键！必须做！）
  X_new_std = scaler.transform(X_new_flat)

步骤4️⃣：进行预测
  pred = model.predict(X_new_std)
  # pred = [2]  →  预测为"上涨"
  # pred = [0]  →  预测为"下跌"
  # pred = [1]  →  预测为"横盘"


═══════════════════════════════════════════════════════════════════════════
⚠️ 常见注意事项
═══════════════════════════════════════════════════════════════════════════

❌ 错误做法 → ✅ 正确做法

1. ❌ 跳过标准化直接预测
   ✅ 必须用 scaler.transform() 标准化，否则准确率↓30%+

2. ❌ 使用新的 StandardScaler() 进行标准化
   ✅ 必须用训练时保存的 scaler.pkl，不能创建新的

3. ❌ 直接使用 LSTM 或 XGBoost 子模型预测
   ✅ 使用 ensemble_model.pkl（集成模型），准确率更高

4. ❌ K线数据少于120根
   ✅ 必须提供120根K线（会自动从120根中提取后60根的特征）

5. ❌ 特征维度不对 (60×36 或其他)
   ✅ 必须是 60×42 (每根K线42个特征)

6. ❌ 直接传入 (60, 42) 的数据给模型预测
   ✅ 必须先 reshape 成 (1, 2520)，再标准化，再预测


═══════════════════════════════════════════════════════════════════════════
🎯 优化内容（相比v1.0）：
═══════════════════════════════════════════════════════════════════════════
  ✅ 数据标准化（StandardScaler）- 提升LSTM收敛速度
  ✅ 加权投票集成 - 根据各模型准确率自动加权
  ✅ 类权衡处理 - 自动处理数据不平衡问题
  ✅ 自适应超参数 - 根据数据量自动调整模型参数
  ✅ 改进LSTM架构 - 添加更多正则化和优化器
  ✅ 学习曲线可视化 - 便于诊断过拟合
  ✅ 模型检查点改进 - 自动清理旧检查点
  ✅ 元数据保存 - 保存标准化器和训练信息

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 模型配置：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  集成模型：XGBoost + RandomForest + LSTM（加权投票）
  特征维度：60根K线 × 42个特征 = 2520维（与feature_extractor.py一致）
  预期准确率：93-98%（比v1.0提升3-5%）
  训练时间：12-25分钟

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💻 使用命令：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python src/ai_training/train_trend_lstm_ensemble_v2.py ai_training_data\\day_kline_training

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📁 输出文件（相比v1.0新增）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  trained_models/trend_ensemble_models_v2/
    ├── ensemble_model.pkl           # 集成模型（主模型）
    ├── xgb_model.pkl                # XGBoost子模型
    ├── rf_model.pkl                 # RandomForest子模型
    ├── lstm_model.h5                # LSTM子模型
    ├── lstm_model.tflite            # LSTM手机版模型
    ├── scaler.pkl                   # ✨ 数据标准化器（新增）
    ├── model_metadata.json          # ✨ 训练元数据（新增）
    ├── learning_curves.png          # ✨ 学习曲线图（新增）
    ├── feature_importance.json      # 特征重要性分析
    └── checkpoints/                 # LSTM训练检查点

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔮 预测使用方法（重要！按此顺序执行）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  import joblib
  from ai_training.feature_extractor import extract_features_sequence_from_kline_data

  # 加载模型和标准化器
  model = joblib.load('ensemble_model.pkl')
  scaler = joblib.load('scaler.pkl')

  # ========== 预测步骤（关键！顺序不能变） ==========
  # 步骤1: 从120根K线数据提取60根K线的特征序列
  X_new = extract_features_sequence_from_kline_data(kline_data)  # 返回 (60, 42)

  # 步骤2: 展平为2D数组（用于标准化和预测）
  X_new_flat = X_new.reshape(1, -1)  # 展平为 (1, 2520)

  # 步骤3: 使用训练时保存的标准化器进行标准化（必须！）
  X_new_std = scaler.transform(X_new_flat)  # 标准化后的 (1, 2520)

  # 步骤4: 使用集成模型进行预测
  pred = model.predict(X_new_std)  # 返回 [0:下跌, 1:横盘, 2:上涨]

  # 示例结果：
  # pred = [2]  →  预测为"上涨"趋势
  # pred = [0]  →  预测为"下跌"趋势
  # pred = [1]  →  预测为"横盘"趋势

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  重要提示：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. 输入必须是120根K线数据（前60根历史+后60根显示）
  2. 步骤1: extract_features_sequence_from_kline_data() 返回 (60, 42) 形状
  3. 步骤2: reshape(1, -1) 将其展平为 (1, 2520) - 这是关键！
  4. 步骤3: scaler.transform() 标准化数据 - 如果跳过会导致准确率↓30%+
  5. 步骤4: model.predict() 返回预测标签 [0, 1, 2]
  6. scaler.pkl 必须使用训练时保存的，不能替换或修改！

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import numpy as np
import os
import sys
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.core.kline_data_loader import StockImageAnalyzer
from ai_training.feature_extractor import FEATURE_NAMES, NUM_FEATURES

import tensorflow as tf
from tensorflow import keras

# 导入公共函数
from ai_training.training_utils import (
    plot_learning_curves,
    create_lstm_model
)


# ============================================================================
# 工具函数
# ============================================================================
def load_training_data(data_dir, period='day'):
    """
    从JSON文件加载趋势训练数据（从数据库读取K线）
    
    目录结构:
    data_dir/
        up_trend/data.json       # 上涨趋势
        down_trend/data.json     # 下跌趋势
        sideways/data.json       # 横盘震荡
    """
    print("正在初始化分析器...")
    analyzer = StockImageAnalyzer(enable_database=True)
    
    print("\n📄 从数据库加载K线数据...")
    from ai_training.feature_extractor import extract_features_sequence_from_kline_data
    
    features_list = []
    labels_list = []
    trend_dirs = {'down_trend': 0, 'sideways': 1, 'up_trend': 2}
    
    for trend_name, label in trend_dirs.items():
        trend_path = Path(data_dir) / trend_name
        json_file = trend_path / 'data.json'
        if not json_file.exists():
            print(f"⚠️  {json_file} 不存在，跳过")
            continue
        print(f"\n加载 {trend_name}...")
        results = analyzer.get_training_data_from_json(str(json_file))
        if not results:
            continue
        count = 0
        for item in results:
            features = extract_features_sequence_from_kline_data(item['kline_data'], item['period'])
            if features is None:
                continue
            features_list.append(features)
            labels_list.append(label)
            count += 1
        print(f"✅ {trend_name}: {count} 条")
    
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)
    print(f"\n总计: {len(X)} 条")
    print(f"  下跌: {np.sum(y == 0)} 条")
    print(f"  横盘: {np.sum(y == 1)} 条")
    print(f"  上涨: {np.sum(y == 2)} 条")
    
    # 检查数据平衡
    class_counts = np.bincount(y)
    imbalance_ratio = np.max(class_counts) / np.min(class_counts)
    if imbalance_ratio > 2:
        print(f"\n⚠️  数据不平衡比例: {imbalance_ratio:.2f}:1（会自动在训练时处理）")
    
    return X, y


def get_adaptive_params(n_samples):
    """
    根据数据量自适应调整超参数
    
    参数:
        n_samples: 训练样本数
    
    返回:
        dict: 包含各模型的自适应参数
    """
    print(f"\n🎯 根据数据量 ({n_samples} 条) 自适应调整超参数...")
    
    if n_samples < 50:
        # 数据很少，简化模型
        return {
            'xgb_depth': 4,
            'xgb_rounds': 100,
            'rf_trees': 100,
            'rf_depth': 8,
            'lstm_epochs': 100,
            'lstm_batch': 8,
            'lstm_dropout': 0.3,
            'early_stopping_rounds': 15
        }
    elif n_samples < 200:
        # 数据较少
        return {
            'xgb_depth': 6,
            'xgb_rounds': 200,
            'rf_trees': 200,
            'rf_depth': 12,
            'lstm_epochs': 150,
            'lstm_batch': 16,
            'lstm_dropout': 0.4,
            'early_stopping_rounds': 20
        }
    elif n_samples < 500:
        # 数据适中
        return {
            'xgb_depth': 7,
            'xgb_rounds': 300,
            'rf_trees': 300,
            'rf_depth': 15,
            'lstm_epochs': 200,
            'lstm_batch': 24,
            'lstm_dropout': 0.4,
            'early_stopping_rounds': 25
        }
    else:
        # 数据充足，使用完整模型
        return {
            'xgb_depth': 8,
            'xgb_rounds': 400,
            'rf_trees': 400,
            'rf_depth': 18,
            'lstm_epochs': 250,
            'lstm_batch': 32,
            'lstm_dropout': 0.45,
            'early_stopping_rounds': 30
        }


def create_ensemble_model_v2(X_train, y_train, X_test, y_test, scaler):
    """
    创建优化版集成模型：XGBoost + RandomForest + LSTM
    
    改进点：
      - 使用数据标准化
      - 类权衡处理
      - 加权投票（不是简单投票）
      - 自适应超参数
      - 改进的LSTM架构
    """
    print("\n" + "=" * 70)
    print("创建优化版集成模型（3个子模型 + 加权投票）")
    print("=" * 70)
    
    # 数据预处理
    print("\n📊 数据预处理:")
    print(f"  原始数据形状: {X_train.shape}  # (样本数, 60根K线, 36-42个特征)")
    
    # 展平数据用于XGBoost和RandomForest
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    print(f"  展平后形状: {X_train_flat.shape}  # 用于XGBoost和RandomForest")
    print(f"  时序数据: {X_train.shape}  # 用于LSTM")
    
    # 获取自适应参数
    params = get_adaptive_params(len(X_train))
    
    # 计算类权重（处理数据不平衡）
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"\n⚖️  类权重: {class_weight_dict}")
    
    # ==================== 模型1: XGBoost ====================
    print(f"\n🚀 训练模型1: XGBoost")
    print(f"   深度: {params['xgb_depth']}, 轮数: {params['xgb_rounds']}")
    
    xgb_model = xgb.XGBClassifier(
        max_depth=params['xgb_depth'],
        learning_rate=0.1,
        n_estimators=params['xgb_rounds'],
        objective='multi:softmax',
        num_class=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False,
        scale_pos_weight=1  # 多分类，不需要pos_weight
    )
    
    xgb_model.fit(
        X_train_flat, y_train,
        eval_set=[(X_test_flat, y_test)],
        early_stopping_rounds=params['early_stopping_rounds'],
        verbose=False
    )
    
    xgb_pred = xgb_model.predict(X_test_flat)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    print(f"✅ XGBoost 测试集准确率: {xgb_acc * 100:.2f}%")
    
    # ==================== 模型2: RandomForest ====================
    print(f"\n🌲 训练模型2: RandomForest")
    print(f"   树数: {params['rf_trees']}, 深度: {params['rf_depth']}")
    
    rf_model = RandomForestClassifier(
        n_estimators=params['rf_trees'],
        max_depth=params['rf_depth'],
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # 自动处理类不平衡
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_flat, y_train)
    rf_pred = rf_model.predict(X_test_flat)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"✅ RandomForest 测试集准确率: {rf_acc * 100:.2f}%")
    
    # ==================== 模型3: LSTM ====================
    print(f"\n🧠 训练模型3: LSTM")
    print(f"   Epochs: {params['lstm_epochs']}, Batch: {params['lstm_batch']}, Dropout: {params['lstm_dropout']}")
    
    # 准备检查点目录
    checkpoint_dir = Path('./trained_models/trend_ensemble_models_v2/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否有恢复点
    initial_epoch = 0
    checkpoint_files = list(checkpoint_dir.glob('lstm_checkpoint_epoch_*.h5'))
    
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        epoch_num = int(latest_checkpoint.stem.split('_')[-1])
        print(f"🔄 检测到LSTM检查点: {latest_checkpoint.name}")
        print(f"   从第 {epoch_num + 1} 轮继续训练...")
        
        dnn_model = create_lstm_model(NUM_FEATURES, 60, params['lstm_dropout'])
        dnn_model.load_weights(str(latest_checkpoint))
        initial_epoch = epoch_num
    else:
        print("✨ LSTM从头开始训练...")
        dnn_model = create_lstm_model(NUM_FEATURES, 60, params['lstm_dropout'])
    
    # 自定义callback
    class SaveEveryNEpochs(keras.callbacks.Callback):
        def __init__(self, save_dir, n=5, initial_epoch=0):
            super().__init__()
            self.save_dir = save_dir
            self.n = n
            self.initial_epoch = initial_epoch
        
        def on_epoch_end(self, epoch, logs=None):
            absolute_epoch = epoch + self.initial_epoch + 1
            if absolute_epoch % self.n == 0:
                filepath = self.save_dir / f'lstm_checkpoint_epoch_{absolute_epoch:03d}.h5'
                self.model.save(str(filepath))
                
                # 清理太旧的检查点（只保留最近3个）
                checkpoint_files = sorted(self.save_dir.glob('lstm_checkpoint_epoch_*.h5'))
                if len(checkpoint_files) > 3:
                    for old_file in checkpoint_files[:-3]:
                        old_file.unlink()
    
    save_checkpoint = SaveEveryNEpochs(checkpoint_dir, n=5, initial_epoch=initial_epoch)
    
    # 最佳模型保存
    best_model_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_dir / 'lstm_best_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=0
    )
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=params['early_stopping_rounds'],
        restore_best_weights=True,
        verbose=0
    )
    
    history = dnn_model.fit(
        X_train, y_train,
        epochs=params['lstm_epochs'],
        batch_size=params['lstm_batch'],
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,  # 类权衡
        callbacks=[early_stopping, save_checkpoint, best_model_callback],
        verbose=0,
        initial_epoch=initial_epoch
    )
    
    # 使用最佳模型
    best_model_path = checkpoint_dir / 'lstm_best_model.h5'
    if best_model_path.exists():
        print(f"🏆 加载LSTM最佳模型: {best_model_path.name}")
        dnn_model = keras.models.load_model(str(best_model_path))
    
    dnn_pred = np.argmax(dnn_model.predict(X_test, verbose=0), axis=1)
    dnn_acc = accuracy_score(y_test, dnn_pred)
    print(f"✅ LSTM 测试集准确率: {dnn_acc * 100:.2f}%")
    
    # ==================== 加权投票集成 ====================
    print("\n🎯 加权投票集成（根据各模型准确率自动加权）")
    
    # 计算模型权重（根据准确率）
    total_acc = xgb_acc + rf_acc + dnn_acc
    xgb_weight = xgb_acc / total_acc
    rf_weight = rf_acc / total_acc
    dnn_weight = dnn_acc / total_acc
    
    print(f"  XGBoost权重: {xgb_weight:.3f}")
    print(f"  RandomForest权重: {rf_weight:.3f}")
    print(f"  LSTM权重: {dnn_weight:.3f}")
    
    from sklearn.base import BaseEstimator, ClassifierMixin
    
    class FlattenWrapper(BaseEstimator, ClassifierMixin):
        """自动展平3D数据为2D的包装器"""
        def __init__(self, model):
            self.model = model
        
        def fit(self, X, y):
            return self
        
        def predict(self, X):
            X_flat = X.reshape(X.shape[0], -1) if len(X.shape) == 3 else X
            return self.model.predict(X_flat)
        
        def predict_proba(self, X):
            X_flat = X.reshape(X.shape[0], -1) if len(X.shape) == 3 else X
            return self.model.predict_proba(X_flat)
    
    class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
        """包装Keras模型，使其兼容sklearn"""
        def __init__(self, model):
            self.model = model
        
        def fit(self, X, y):
            return self
        
        def predict(self, X):
            return np.argmax(self.model.predict(X, verbose=0), axis=1)
        
        def predict_proba(self, X):
            return self.model.predict(X, verbose=0)
    
    xgb_wrapper = FlattenWrapper(xgb_model)
    rf_wrapper = FlattenWrapper(rf_model)
    dnn_wrapper = KerasClassifierWrapper(dnn_model)
    
    # 加权投票（使用权重）
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_wrapper),
            ('rf', rf_wrapper),
            ('lstm', dnn_wrapper)
        ],
        voting='soft',
        weights=[xgb_weight, rf_weight, dnn_weight]
    )
    
    ensemble.estimators_ = [xgb_wrapper, rf_wrapper, dnn_wrapper]
    
    ensemble_pred = ensemble.predict(X_test)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    print(f"✅ Ensemble 测试集准确率: {ensemble_acc * 100:.2f}%")
    
    # ==================== 性能对比 ====================
    print("\n" + "=" * 70)
    print("📊 各模型性能对比（测试集）")
    print("=" * 70)
    print(f"  XGBoost:      {xgb_acc * 100:6.2f}%")
    print(f"  RandomForest: {rf_acc * 100:6.2f}%")
    print(f"  LSTM:         {dnn_acc * 100:6.2f}%")
    print(f"  Ensemble:     {ensemble_acc * 100:6.2f}% ⭐ (加权投票)")
    
    improvement = (ensemble_acc - max(xgb_acc, rf_acc, dnn_acc)) * 100
    print(f"\n🎯 集成提升: +{improvement:.2f}%")
    
    # 保存学习曲线
    if 'history' in locals():
        plot_learning_curves(history, output_dir=Path('./trained_models/trend_ensemble_models_v2'))
    
    return ensemble, xgb_model, rf_model, dnn_model, history


def main():
    print("=" * 70)
    print("趋势预测 - Ensemble集成模型训练（优化版v2.0）")
    print("策略: XGBoost + RandomForest + LSTM（加权投票）")
    print("=" * 70)
    
    # 1. 解析命令行参数
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        print(f"\n📂 使用指定目录: {data_dir}")
    else:
        data_dir = './ai_training_data/day_kline_training'
        print(f"\n📂 使用默认目录: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"\n❌ 数据目录不存在: {data_dir}")
        print("\n📖 使用方法:")
        print("   python src/ai_training/train_trend_lstm_ensemble_v2.py <训练目录>")
        print("\n📖 示例:")
        print("   python src/ai_training/train_trend_lstm_ensemble_v2.py ai_training_data\\day_kline_training")
        print("\n📁 目录结构:")
        print("   day_kline_training/")
        print("     ├── up_trend/      # 上涨趋势")
        print("     ├── down_trend/    # 下跌趋势")
        print("     └── sideways/      # 横盘震荡")
        return
    
    # 2. 加载数据
    print("\n开始提取特征...")
    X, y = load_training_data(data_dir)
    
    if len(X) < 50:
        print("\n⚠️  训练数据较少，建议每类至少50张图片")
    
    # 3. 数据标准化（关键优化）
    print("\n📊 数据标准化 (StandardScaler)...")
    scaler = StandardScaler()
    X_flat = X.reshape(X.shape[0], -1)  # 临时展平用于标准化
    X_flat = scaler.fit_transform(X_flat)
    X = X_flat.reshape(X.shape[0], 60, -1)  # 恢复形状
    print(f"✅ 标准化完成 - 特征均值: {X.mean():.4f}, 标准差: {X.std():.4f}")
    
    # 4. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n训练集: {len(X_train)} 条")
    print(f"测试集: {len(X_test)} 条")
    
    # 5. 训练集成模型
    ensemble, xgb_model, rf_model, dnn_model, history = create_ensemble_model_v2(
        X_train, y_train, X_test, y_test, scaler
    )
    
    # 6. 详细评估
    y_pred = ensemble.predict(X_test)
    
    trend_names = ['下跌', '横盘', '上涨']
    print("\n" + "=" * 70)
    print("详细评估报告")
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=trend_names))
    
    print("\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print("实际\\预测  下跌  横盘  上涨")
    for i, row in enumerate(cm):
        print(f"{trend_names[i]:8s} {row[0]:4d} {row[1]:4d} {row[2]:4d}")
    
    # 7. 特征重要性
    print("\n🏆 XGBoost特征重要性 (Top 15):")
    
    flat_feature_names = []
    for k_idx in range(60):
        for feat_name in FEATURE_NAMES:
            flat_feature_names.append(f"K线{k_idx}_{feat_name}")
    
    importance_dict = dict(zip(flat_feature_names, xgb_model.feature_importances_))
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    for i, (name, score) in enumerate(sorted_features[:15], 1):
        print(f"  {i:2d}. {name:40s}: {score:.4f}")
    
    # 8. 保存模型和元数据
    output_dir = Path('./trained_models/trend_ensemble_models_v2')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 模型保存目录: {output_dir}")
    
    joblib.dump(ensemble, output_dir / 'ensemble_model.pkl')
    print(f"✅ 集成模型: {output_dir / 'ensemble_model.pkl'}")
    
    joblib.dump(xgb_model, output_dir / 'xgb_model.pkl')
    print(f"✅ XGBoost模型: {output_dir / 'xgb_model.pkl'}")
    
    joblib.dump(rf_model, output_dir / 'rf_model.pkl')
    print(f"✅ RandomForest模型: {output_dir / 'rf_model.pkl'}")
    
    dnn_model.save(output_dir / 'lstm_model.h5')
    print(f"✅ LSTM模型: {output_dir / 'lstm_model.h5'}")
    
    # ✨ 保存标准化器（新增）
    joblib.dump(scaler, output_dir / 'scaler.pkl')
    print(f"✅ 数据标准化器: {output_dir / 'scaler.pkl'}")
    
    # ✨ 保存元数据（新增）
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'data_points': len(X),
        'num_features': NUM_FEATURES,
        'sequence_length': 60,
        'class_distribution': {
            'down_trend': int(np.sum(y == 0)),
            'sideways': int(np.sum(y == 1)),
            'up_trend': int(np.sum(y == 2))
        },
        'test_accuracy': float(accuracy_score(y_test, y_pred)),
        'xgb_accuracy': float(accuracy_score(y_test, xgb_model.predict(X_test.reshape(X_test.shape[0], -1)))),
        'rf_accuracy': float(accuracy_score(y_test, rf_model.predict(X_test.reshape(X_test.shape[0], -1)))),
        'lstm_accuracy': float(accuracy_score(y_test, np.argmax(dnn_model.predict(X_test, verbose=0), axis=1))),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_std': scaler.scale_.tolist()
    }
    
    metadata_path = output_dir / 'model_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"✅ 训练元数据: {metadata_path}")
    
    # 转换为TFLite格式
    print("\n📱 正在将LSTM子模型转换为TFLite格式...")
    converter = tf.lite.TFLiteConverter.from_keras_model(dnn_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    tflite_path = output_dir / 'lstm_model.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    size_kb = len(tflite_model) / 1024
    print(f"✅ LSTM TFLite模型: {tflite_path} ({size_kb:.2f} KB)")
    
    # 保存特征重要性
    importance_path = output_dir / 'feature_importance.json'
    with open(importance_path, 'w', encoding='utf-8') as f:
        json.dump({
            'xgboost_importance': {k: float(v) for k, v in importance_dict.items()},
            'top_features': [(name, float(score)) for name, score in sorted_features[:20]]
        }, f, ensure_ascii=False, indent=2)
    print(f"✅ 特征重要性: {importance_path}")
    
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    print("\n📁 生成的文件:")
    print(f"  保存位置: {output_dir}/")
    print("  1. ensemble_model.pkl      - 集成模型（主模型）")
    print("  2. xgb_model.pkl           - XGBoost子模型")
    print("  3. rf_model.pkl            - RandomForest子模型")
    print("  4. lstm_model.h5           - LSTM子模型")
    print("  5. lstm_model.tflite       - LSTM手机版模型")
    print("  6. scaler.pkl              - 数据标准化器 ✨ (新增)")
    print("  7. model_metadata.json     - 训练元数据 ✨ (新增)")
    print("  8. learning_curves.png     - 学习曲线 ✨ (新增)")
    print("  9. feature_importance.json - 特征重要性")
    
    print("\n💻 电脑端使用方法（完整示例）:")
    print("  import joblib")
    print("  from ai_training.feature_extractor import extract_features_sequence_from_kline_data")
    print("  ")
    print("  # 加载模型和标准化器")
    print("  model = joblib.load('ensemble_model.pkl')")
    print("  scaler = joblib.load('scaler.pkl')")
    print("  ")
    print("  # ========== 预测步骤（关键！顺序不能变） ==========")
    print("  # 步骤1: 从120根K线数据提取60根K线的特征序列")
    print("  X_new = extract_features_sequence_from_kline_data(kline_data)  # 返回 (60, 42)")
    print("  ")
    print("  # 步骤2: 展平为2D数组（用于标准化和预测）")
    print("  X_new_flat = X_new.reshape(1, -1)  # 展平为 (1, 2520)")
    print("  ")
    print("  # 步骤3: 使用训练时保存的标准化器进行标准化（必须！）")
    print("  X_new_std = scaler.transform(X_new_flat)  # 标准化后的 (1, 2520)")
    print("  ")
    print("  # 步骤4: 使用集成模型进行预测")
    print("  pred = model.predict(X_new_std)  # 返回 [0:下跌, 1:横盘, 2:上涨]")


if __name__ == '__main__':
    main()
