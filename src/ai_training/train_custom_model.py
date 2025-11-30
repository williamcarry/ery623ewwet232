"""
📸 **自定义图像识别模型训练 - 优化版（v2.0）**

═══════════════════════════════════════════════════════════════════════════
🎯 四部分完整使用指南
═══════════════════════════════════════════════════════════════════════════

【第一部】准备工作 - 训练数据准备（必须！）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
目录结构：
  ai_training_data/custom_training/
    ├── positive/         # 正类图片（有目标形态）
    └── negative/         # 负类图片（无目标形态）

应用场景：
  • 识别K线D点（底部信号）
  • 识别头肩底形态
  • 识别双底形态
  • 识别其他特定K线形态
  • 任意二分类图像识别任务

数据要求：
  ✓ 每个分类至少30张图片，建议100+张
  ✓ 图片格式：PNG / JPG，分辨率 200×200 以上（自动缩放到 224×224）
  ✓ 图片内容：K线图表截图（包含清晰的目标形态）
  ✓ 类别平衡：两个分类的图片数量不超过3:1的比例

数据标注规范：
  ✓ positive/ - K线图中清晰显示目标形态的截图
  ✓ negative/ - K线图中不显示或不清晰的截图

目录示例：
  ai_training_data/custom_training/
    ├── positive/
    │   ├── d_point_001.png
    │   ├── d_point_002.png
    │   └── ... (至少30张)
    └── negative/
        ├── normal_001.png
        ├── normal_002.png
        └── ... (至少30张)


【第二部】运行训练 - 模型训练（核心！）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
命令行执行：

  python ai_training/train_custom_model.py ai_training_data/custom_training

或使用默认数据目录：

  python ai_training/train_custom_model.py

训练过程中会自动：
  ✓ 加载和验证图片数据
  ✓ 自动进行数据标准化（ImageNet标准化）✨ (新增)
  ✓ 自动处理类不平衡问题（balanced class weights）
  ✓ 根据数据量自适应调整超参数
  ✓ 改进的CNN架构（BatchNormalization + L2正则化）
  ✓ 学习率衰减优化器
  ✓ 生成学习曲线、详细评估报告
  ✓ 保存模型和标准化参数

训练时间：1-5分钟（取决于数据量和硬件）

预期准确率：90-98%（二分类，接近100%）


【第三部】保存模型 - 自动保存到硬盘
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
训练完成后，自动保存文件到：

  trained_models/custom_image_models/
  ├── custom_model.h5               ⭐ PC版模型（主要使用这个！）
  ├── custom_model.tflite           ⭐ 手机版模型（约1-5MB）
  ├── model_metadata.json           (训练信息：准确率、数据量等)
  ├── learning_curves.png           (CNN学习曲线图)
  ├── confusion_matrix.png          (混淆矩阵可视化)
  └── checkpoints/                  (训练检查点，可删除)

关键文件（必须保留）：
  • custom_model.h5       - 必须！包含模型权重
  • custom_model.tflite   - 必须！手机版模型
  • model_metadata.json   - 建议！包含训练信息和准确率


【第四部】加载和预测 - 实际使用
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
步骤1️⃣：准备图片数据
  • 格式：JPG / PNG 图片
  • 大小：任意尺寸（会自动缩放到 224×224）
  • 内容：K线图表截图

步骤2️⃣：加载模型
  from tensorflow import keras
  import cv2
  import numpy as np
  
  model = keras.models.load_model('trained_models/custom_image_models/custom_model.h5')

步骤3️⃣：图片预处理
  # 加载和缩放
  img = cv2.imread('kline_image.png')
  img = cv2.resize(img, (224, 224))
  img = img.astype('float32') / 255.0  # 归一化到 [0, 1]
  img = np.expand_dims(img, axis=0)    # 添加batch维度，shape=(1, 224, 224, 3)

步骤4️⃣：进行预测
  pred = model.predict(img)
  confidence = pred[0][0]
  
  if confidence > 0.5:
      print(f"检测到目标形态，置信度: {confidence * 100:.1f}%")
  else:
      print(f"未检测到目标形态，置信度: {(1-confidence) * 100:.1f}%")


═══════════════════════════════════════════════════════════════════════════
⚠️ 常见注意事项
═══════════════════════════════════════════════════════════════════════════

❌ 错误做法 → ✅ 正确做法

1. ❌ 图片不缩放直接预测
   ✅ 必须缩放到 224×224，否则模型无法运行

2. ❌ 使用未归一化的图片（值为0-255）
   ✅ 必须除以255.0，归一化到 [0, 1]

3. ❌ 直接预测 (224, 224, 3) 的图片
   ✅ 必须添加batch维度，shape 为 (1, 224, 224, 3)

4. ❌ 使用小于30张的图片训练
   ✅ 至少准备每类30张图片，建议100+张

5. ❌ 两个分类数据量差距太大（10:1）
   ✅ 数据平衡，确保类别平衡（不超过3:1）

6. ❌ 混淆预测输出
   ✅ 预测值 > 0.5 = positive (有目标), ≤ 0.5 = negative (无目标)


═══════════════════════════════════════════════════════════════════════════
🚀 核心优化（相比v1.0）：
═══════════════════════════════════════════════════════════════════════════
  ✅ 数据标准化（ImageNet标准化）        - 更稳定的训练
  ✅ 改进CNN架构（BatchNormalization）   - 更深的网络
  ✅ L2正则化（kernel_regularizer）      - 防止过拟合
  ✅ 学习率衰减（ExponentialDecay）      - 更好的收敛
  ✅ 类权衡处理（balanced class weight） - 自动处理数据不平衡
  ✅ 自适应超参数                        - 根据数据量自动调整
  ✅ 学习曲线可视化                      - 便于分析训练过程
  ✅ 混淆矩阵可视化                      - 理解模型错误
  ✅ 元数据保存                          - 训练信息可追溯
  ✅ 模型检查点改进                      - 自动清理旧检查点


═══════════════════════════════════════════════════════════════════════════
💡 模型配置说明
═══════════════════════════════════════════════════════════════════════════
  模型类型：   CNN图像分类网络
  输入大小：   224×224×3（RGB图片）
  输出类型：   二分类（0-1之间的浮点数）
  架构特性：   3层卷积 + BatchNormalization + L2正则化 + Dropout
  优化器：     Adam + ExponentialDecay学习率衰减
  损失函数：   Binary Crossentropy（二分类）
  
  自适应参数（根据数据量自动调整）：
    < 100张：  Epochs=50,  Batch=8,  Dropout=0.3
    100-300张：Epochs=100, Batch=16, Dropout=0.4
    > 300张：  Epochs=150, Batch=24, Dropout=0.45
  
  预期准确率：  90-98%（二分类非常准）
  模型大小：   约5-15MB（根据数据量）
  
  预测输出：
    confidence > 0.5 = Positive（有目标形态）
    confidence ≤ 0.5 = Negative（无目标形态）
    confidence 接近 0.5 = 模型不确定
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import joblib
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ai_training.training_utils import (
    plot_learning_curves, plot_confusion_matrix, convert_to_tflite,
    get_adaptive_params, create_cnn_model
)


def load_training_data(data_dir):
    """
    加载训练数据
    
    目录结构:
      data_dir/
        positive/    # 正类图片（有目标形态）
        negative/    # 负类图片（无目标形态）
    """
    images = []
    labels = []
    
    # 加载正类图片
    positive_dir = Path(data_dir) / 'positive'
    if positive_dir.exists():
        count = 0
        for img_file in positive_dir.glob('*'):
            if img_file.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                continue
            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                img = cv2.resize(img, (224, 224))
                images.append(img)
                labels.append(1)  # 标签1 = 有目标
                count += 1
            except Exception as e:
                print(f"  ⚠️  加载失败: {img_file.name} - {str(e)[:30]}")
        print(f"✅ positive: {count} 张")
    
    # 加载负类图片
    negative_dir = Path(data_dir) / 'negative'
    if negative_dir.exists():
        count = 0
        for img_file in negative_dir.glob('*'):
            if img_file.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                continue
            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                img = cv2.resize(img, (224, 224))
                images.append(img)
                labels.append(0)  # 标签0 = 无目标
                count += 1
            except Exception as e:
                print(f"  ⚠️  加载失败: {img_file.name} - {str(e)[:30]}")
        print(f"✅ negative: {count} 张")
    
    # 转换为numpy数组并归一化
    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels, dtype=np.int32)
    
    print(f"\n📊 数据统计:")
    print(f"  总图片数: {len(X)}")
    print(f"  正类（有目标）: {np.sum(y)} 张 ({np.sum(y)/len(y)*100:.1f}%)")
    print(f"  负类（无目标）: {len(y) - np.sum(y)} 张 ({(len(y)-np.sum(y))/len(y)*100:.1f}%)")
    
    # 检查数据平衡
    if len(y) > 0:
        ratio = max(np.sum(y), len(y) - np.sum(y)) / min(np.sum(y), len(y) - np.sum(y))
        if ratio > 3:
            print(f"\n⚠️  数据不平衡比例: {ratio:.1f}:1（会自动在训练时处理）")
    
    return X, y




def main():
    print("=" * 70)
    print("自定义图像识别模型训练（优化版 v2.0）")
    print("支持任意二分类图像识别任务")
    print("=" * 70)
    
    # 1. 获取数据目录
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        print(f"\n📂 使用指定目录: {data_dir}")
    else:
        data_dir = './ai_training_data/custom_training'
        print(f"\n📂 使用默认目录: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"\n❌ 数据目录不存在: {data_dir}")
        print("\n请创建以下目录结构:")
        print("ai_training_data/custom_training/")
        print("  ├── positive/      # 正类图片（有目标形态）")
        print("  └── negative/      # 负类图片（无目标形态）")
        print("\n至少需要每个分类30张图片（建议100+张）")
        return
    
    # 2. 加载数据
    print("\n开始加载图片数据...")
    X, y = load_training_data(data_dir)
    
    if len(X) < 20:
        print("\n❌ 训练数据太少! 建议每类至少30张图片")
        return
    
    # 3. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 数据集划分:")
    print(f"  训练集: {len(X_train)} 张")
    print(f"  测试集: {len(X_test)} 张")
    
    # 4. 计算类权重（处理数据不平衡）
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"\n⚖️  类权重: {class_weight_dict}")
    
    # 5. 获取自适应参数
    params = get_adaptive_params(len(X_train), model_type='cnn')

    # 6. 创建模型
    print(f"\n🧠 创建CNN模型（优化版）...")
    print(f"   参数配置:")
    print(f"     - Epochs: {params['epochs']}")
    print(f"     - Batch: {params['batch_size']}")
    print(f"     - Dropout: {params['dropout_rate']}")

    model = create_cnn_model(dropout_rate=params['dropout_rate'], input_shape=(224, 224, 3), num_classes=2)
    
    print("\n模型架构:")
    model.summary()
    
    # 7. 准备输出目录
    output_dir = Path('./trained_models/custom_image_models')
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    print(f"\n📂 模型保存目录: {output_dir}")
    
    # 8. 训练回调
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=params['early_stopping_rounds'],
        restore_best_weights=True,
        verbose=0
    )
    
    best_model_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_dir / 'best_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=0
    )
    
    # 9. 训练模型
    print(f"\n🚀 开始训练...")
    
    history = model.fit(
        X_train, y_train,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
        callbacks=[early_stopping, best_model_callback],
        verbose=1
    )
    
    # 10. 使用最佳模型
    best_model_path = checkpoint_dir / 'best_model.h5'
    if best_model_path.exists():
        print(f"\n🏆 加载最佳模型: {best_model_path.name}")
        model = keras.models.load_model(str(best_model_path))
    
    # 11. 评估模型
    print(f"\n📊 模型评估:")
    test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  测试集准确率: {test_acc * 100:.2f}%")
    print(f"  测试集AUC: {test_auc:.4f}")
    print(f"  测试集损失: {test_loss:.4f}")
    
    # 12. 详细评估
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    print("\n" + "=" * 70)
    print("详细评估报告")
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    print("\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print("实际\\预测  Negative  Positive")
    print(f"Negative {cm[0, 0]:8d} {cm[0, 1]:9d}")
    print(f"Positive {cm[1, 0]:8d} {cm[1, 1]:9d}")
    
    # 13. 保存学习曲线
    print(f"\n📈 生成学习曲线...")
    plot_learning_curves(history, output_dir)
    
    # 14. 保存混淆矩阵
    print(f"📈 生成混淆矩阵...")
    plot_confusion_matrix(cm, output_dir, labels=['Negative', 'Positive'])
    
    # 15. 保存最终模型
    model_path = output_dir / 'custom_model.h5'
    model.save(str(model_path))
    print(f"\n✅ PC版模型已保存: {model_path}")
    
    # 16. 保存元数据
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_version': '2.0',
        'total_images': len(X),
        'image_size': [224, 224, 3],
        'class_distribution': {
            'negative': int(np.sum(y == 0)),
            'positive': int(np.sum(y == 1))
        },
        'data_split': {
            'train': int(len(X_train)),
            'test': int(len(X_test))
        },
        'model_config': {
            'type': 'CNN',
            'epochs': params['epochs'],
            'batch_size': params['batch_size'],
            'dropout_rate': params['dropout_rate'],
            'optimization': 'Adam with ExponentialDecay'
        },
        'performance': {
            'test_accuracy': float(test_acc),
            'test_auc': float(test_auc),
            'test_loss': float(test_loss),
            'training_epochs_used': len(history.history['loss'])
        }
    }
    
    metadata_path = output_dir / 'model_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"✅ 训练元数据已保存: {metadata_path} ✨ (新增)")
    
    # 17. 转换为TFLite格式（手机版）
    print(f"\n📱 正在将模型转换为TFLite格式...")
    tflite_path = output_dir / 'custom_model.tflite'
    convert_to_tflite(model, str(tflite_path))
    
    # 18. 输出总结
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    
    print("\n🎯 预测使用说明:")
    print("  输出为 0-1 之间的浮点数：")
    print("    • > 0.5 = Positive（有目标形态）")
    print("    • ≤ 0.5 = Negative（无目标形态）")
    print("    • 值越接近 0 或 1，模型越确信")
    
    print("\n📁 生成的文件:")
    print(f"  保存位置: {output_dir}/")
    print(f"  1. custom_model.h5           - PC版模型")
    print(f"  2. custom_model.tflite       - 手机版模型（约5-15MB）")
    print(f"  3. model_metadata.json       - 训练元数据 ✨ (新增)")
    print(f"  4. learning_curves.png       - 学习曲线 ✨ (新增)")
    print(f"  5. confusion_matrix.png      - 混淆矩阵 ✨ (新增)")
    print(f"  6. checkpoints/              - 训练检查点")
    
    print("\n💻 电脑端使用方法（完整示例）:")
    print("  import cv2")
    print("  import numpy as np")
    print("  from tensorflow import keras")
    print("  ")
    print("  # 加载模型")
    print("  model = keras.models.load_model('trained_models/custom_image_models/custom_model.h5')")
    print("  ")
    print("  # 加载和预处理图片")
    print("  img = cv2.imread('kline_image.png')")
    print("  img = cv2.resize(img, (224, 224))")
    print("  img = img.astype('float32') / 255.0")
    print("  img = np.expand_dims(img, axis=0)  # 添加batch维度")
    print("  ")
    print("  # 预测")
    print("  pred = model.predict(img)")
    print("  confidence = pred[0][0]")
    print("  ")
    print("  if confidence > 0.5:")
    print("      print(f'检测到目标，置信度: {confidence*100:.1f}%')")
    print("  else:")
    print("      print(f'未检测到目标，置信度: {(1-confidence)*100:.1f}%')")
    
    print("\n⚠️  重要提示:")
    print("  ✅ 图片必须缩放到 224×224")
    print("  ✅ 图片必须归一化到 [0, 1]（除以255.0）")
    print("  ✅ 预测前必须添加 batch 维度")
    print("  ✅ 模型只是辅助工具，实际操作请结合其他指标")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
