"""
共享训练工具模块 - 提供所有训练脚本通用的函数

包含：
  • 图表绘制函数
  • 模型转换函数
  • 超参数自适应函数
  • 通用模型创建函数
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib


# ============================================================================
# 绘图函数
# ============================================================================
def plot_learning_curves(history, output_dir):
    """
    绘制LSTM/CNN学习曲线
    
    参数:
        history: keras训练历史对象
        output_dir: 输出目录
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 精度曲线
        ax1.plot(history.history['accuracy'], label='Train Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Learning Curves - Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # 损失曲线
        ax2.plot(history.history['loss'], label='Train Loss')
        ax2.plot(history.history['val_loss'], label='Val Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Learning Curves - Loss')
        ax2.legend()
        ax2.grid(True)
        
        # 保存图表
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_dir / 'learning_curves.png', dpi=100)
        plt.close()
        print(f"✅ 学习曲线已保存: {output_dir / 'learning_curves.png'}")
    except Exception as e:
        print(f"⚠️  绘制学习曲线失败: {e}")


def plot_confusion_matrix(cm, output_dir, labels=None):
    """
    绘制混淆矩阵
    
    参数:
        cm: 混淆矩阵（来自sklearn.metrics.confusion_matrix）
        output_dir: 输出目录
        labels: 类别标签（可选）
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 归一化混淆矩阵
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        im = ax.imshow(cm_normalized, cmap='Blues', aspect='auto')
        
        # 设置标签
        if labels is None:
            labels = [f'Class {i}' for i in range(len(cm))]
        
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        # 添加数值
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2%})',
                             ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=100)
        plt.close()
        print(f"✅ 混淆矩阵已保存: {output_dir / 'confusion_matrix.png'}")
    except Exception as e:
        print(f"⚠️  绘制混淆矩阵失败: {e}")


# ============================================================================
# 模型转换函数
# ============================================================================
def convert_to_tflite(model, output_path):
    """
    将Keras模型转换为TensorFlow Lite格式
    
    参数:
        model: Keras模型
        output_path: 输出路径
    
    返回:
        Path: 输出文件路径
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    # 确保目录存在
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_mb = len(tflite_model) / 1024 / 1024
    size_kb = len(tflite_model) / 1024
    
    if size_mb > 1:
        size_str = f"{size_mb:.2f} MB"
    else:
        size_str = f"{size_kb:.2f} KB"
    
    print(f"✅ TFLite模型已保存: {output_path} ({size_str})")
    
    return output_path


# ============================================================================
# 超参数自适应函数
# ============================================================================
def get_adaptive_params(n_samples, model_type='lstm'):
    """
    根据数据量自适应调整超参数
    
    参数:
        n_samples: 训练样本数
        model_type: 模型类型 ('lstm', 'cnn', 'xgb', 'rf')
    
    返回:
        dict: 包含自适应超参数
    """
    print(f"\n🎯 根据数据量 ({n_samples} 条) 自适应调整超参数...")
    
    if model_type in ['lstm', 'gru', 'transformer']:
        # LSTM/GRU/Transformer参数
        if n_samples < 50:
            return {
                'epochs': 100,
                'batch_size': 8,
                'dropout_rate': 0.3,
                'early_stopping_rounds': 15
            }
        elif n_samples < 200:
            return {
                'epochs': 150,
                'batch_size': 16,
                'dropout_rate': 0.4,
                'early_stopping_rounds': 20
            }
        elif n_samples < 500:
            return {
                'epochs': 200,
                'batch_size': 24,
                'dropout_rate': 0.4,
                'early_stopping_rounds': 25
            }
        else:
            return {
                'epochs': 250,
                'batch_size': 32,
                'dropout_rate': 0.45,
                'early_stopping_rounds': 30
            }
    
    elif model_type == 'cnn':
        # CNN参数
        if n_samples < 100:
            return {
                'epochs': 50,
                'batch_size': 8,
                'dropout_rate': 0.3,
                'early_stopping_rounds': 10
            }
        elif n_samples < 300:
            return {
                'epochs': 100,
                'batch_size': 16,
                'dropout_rate': 0.4,
                'early_stopping_rounds': 15
            }
        else:
            return {
                'epochs': 150,
                'batch_size': 24,
                'dropout_rate': 0.45,
                'early_stopping_rounds': 20
            }
    
    elif model_type in ['xgb', 'rf']:
        # XGBoost/RandomForest参数
        if n_samples < 50:
            return {
                'xgb_depth': 4,
                'xgb_rounds': 100,
                'rf_trees': 100,
                'rf_depth': 8,
                'early_stopping_rounds': 15
            }
        elif n_samples < 200:
            return {
                'xgb_depth': 6,
                'xgb_rounds': 200,
                'rf_trees': 200,
                'rf_depth': 12,
                'early_stopping_rounds': 20
            }
        elif n_samples < 500:
            return {
                'xgb_depth': 7,
                'xgb_rounds': 300,
                'rf_trees': 300,
                'rf_depth': 15,
                'early_stopping_rounds': 25
            }
        else:
            return {
                'xgb_depth': 8,
                'xgb_rounds': 400,
                'rf_trees': 400,
                'rf_depth': 18,
                'early_stopping_rounds': 30
            }


# ============================================================================
# 模型创建函数
# ============================================================================
def create_lstm_model(sequence_length, num_features, num_classes=3, dropout_rate=0.4):
    """
    创建LSTM时序模型（优化版）
    
    参数:
        sequence_length: 时序长度（通常60）
        num_features: 特征数量（通常42或19）
        num_classes: 分类数量（2或3）
        dropout_rate: Dropout比例
    
    返回:
        keras.Model: LSTM模型
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(sequence_length, num_features)),
        
        # 第一层LSTM
        keras.layers.LSTM(
            256,
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l2(0.001)
        ),
        keras.layers.LayerNormalization(),
        keras.layers.Dropout(dropout_rate),
        
        # 第二层LSTM
        keras.layers.LSTM(
            128,
            return_sequences=False,
            kernel_regularizer=keras.regularizers.l2(0.001)
        ),
        keras.layers.LayerNormalization(),
        keras.layers.Dropout(dropout_rate - 0.1),
        
        # 全连接层
        keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.LayerNormalization(),
        keras.layers.Dropout(dropout_rate - 0.1),
        
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)),
        keras.layers.Dropout(dropout_rate - 0.2),
        
        keras.layers.Dense(32, activation='relu'),
        
        # 输出层
        keras.layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    # 使用学习率衰减的优化器
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True
    )
    
    loss_fn = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    return model


def create_gru_model(sequence_length, num_features, num_classes=3, dropout_rate=0.4):
    """
    创建GRU时序模型（优化版，比LSTM更快）
    
    参数:
        sequence_length: 时序长度
        num_features: 特征数量
        num_classes: 分类数量
        dropout_rate: Dropout比例
    
    返回:
        keras.Model: GRU模型
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(sequence_length, num_features)),
        
        # 第一层GRU
        keras.layers.GRU(
            256,
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l2(0.001)
        ),
        keras.layers.LayerNormalization(),
        keras.layers.Dropout(dropout_rate),
        
        # 第二层GRU
        keras.layers.GRU(
            128,
            return_sequences=False,
            kernel_regularizer=keras.regularizers.l2(0.001)
        ),
        keras.layers.LayerNormalization(),
        keras.layers.Dropout(dropout_rate - 0.1),
        
        keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.LayerNormalization(),
        keras.layers.Dropout(dropout_rate - 0.1),
        
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)),
        keras.layers.Dropout(dropout_rate - 0.2),
        
        keras.layers.Dense(32, activation='relu'),
        
        keras.layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True
    )
    
    loss_fn = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    return model


def create_transformer_model(sequence_length, num_features, num_classes=3, dropout_rate=0.4):
    """
    创建Transformer模型（优化版，最先进但需要更多数据）
    
    参数:
        sequence_length: 时序长度
        num_features: 特征数量
        num_classes: 分类数量
        dropout_rate: Dropout比例
    
    返回:
        keras.Model: Transformer模型
    """
    inputs = keras.layers.Input(shape=(sequence_length, num_features))
    
    # Multi-Head Self-Attention
    attention_output = keras.layers.MultiHeadAttention(
        num_heads=4,
        key_dim=32,
        dropout=dropout_rate
    )(inputs, inputs)
    
    attention_output = keras.layers.LayerNormalization()(attention_output + inputs)
    
    # Feed Forward Network
    ffn_output = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(attention_output)
    ffn_output = keras.layers.Dropout(dropout_rate)(ffn_output)
    ffn_output = keras.layers.Dense(num_features)(ffn_output)
    
    ffn_output = keras.layers.LayerNormalization()(ffn_output + attention_output)
    
    # 全局平均池化
    pooled = keras.layers.GlobalAveragePooling1D()(ffn_output)
    
    # 分类头
    x = keras.layers.Dense(64, activation='relu')(pooled)
    x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True
    )
    
    loss_fn = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    return model


def create_cnn_model(dropout_rate=0.4, input_shape=(224, 224, 3), num_classes=2):
    """
    创建CNN图像分类模型（优化版）
    
    参数:
        dropout_rate: Dropout比例
        input_shape: 输入形状（高, 宽, 通道数）
        num_classes: 分类数量（通常为2）
    
    返回:
        keras.Model: CNN模型
    """
    model = keras.Sequential([
        # 输入层
        keras.layers.Input(shape=input_shape),
        
        # 第一层卷积块
        keras.layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(dropout_rate),
        
        # 第二层卷积块
        keras.layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(dropout_rate),
        
        # 第三层卷积块
        keras.layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(dropout_rate),
        
        # 全局平均池化
        keras.layers.GlobalAveragePooling2D(),
        
        # 全连接层
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(dropout_rate - 0.1),
        
        keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)),
        keras.layers.Dropout(dropout_rate - 0.15),
        
        # 输出层（二分类）
        keras.layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    # 使用学习率衰减的优化器
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True
    )
    
    loss_fn = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    return model


# ============================================================================
# 数据保存函数
# ============================================================================
def save_scaler(scaler, output_path):
    """保存StandardScaler"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_path)
    print(f"✅ 数据标准化器已保存: {output_path}")


def load_scaler(scaler_path):
    """加载StandardScaler"""
    return joblib.load(scaler_path)
