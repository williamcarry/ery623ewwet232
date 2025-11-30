"""
⚠️ 警告：此文件已废弃，仅作为备份保留！
==========================================

此文件是43特征版本的旧实现，包含硬编码阈值，不适用于不同价位的股票。
已被 feature_extractor-bak.py (30特征版本) 替代。

主要问题：
1. 硬编码阈值（如±0.5、0.1、0.3等）无法适应不同价位股票
2. 大量独热编码导致信息损失
3. MA25趋势斜率使用绝对值判断，不科学

请使用 feature_extractor-bak.py (30特征版本)！
==========================================
"""
"""
AI训练特征提取器 - 统一特征定义

作用：集中管理所有AI训练使用的特征
优势：修改特征时只需改这一个文件，所有训练脚本自动同步

特征总数：43个
"""
import numpy as np


# ============================================================================
# 特征名称定义（用于可解释性分析）
# ============================================================================
FEATURE_NAMES = [
    # 1. 价格均线特征（11个）
    'MA5价格比例', 'MA25价格比例',  # 相对于收盘价的比例
    '价格>MA5', '价格>MA25',
    'MA5>MA25',  # MA5均线在MA25均线上方（均线金叉状态）
    'MA25趋势向上', 'MA25趋势横盘', 'MA25趋势向下',  # MA25趋势方向（独热编码）
    'K线收阳',  # K线收阳线状态
    '价格相对MA25强度',  # (Close - MA25) / MA25
    'K线在MA25上方',  # K线与MA25位置关系
    
    # 2. MACD特征（19个）
    'DIF比例', 'DEA比例',  # 相对于收盘价的比例
    'MACD柱比例',  # MACD / Close
    'DIF-DEA距离比例',  # |DIF - DEA| / Close
    'MACD零轴上方', 'MACD零轴下方', 'MACD零轴附近',  # 零轴位置（独热）
    'MACD金叉', 'MACD死叉', 'MACD纠缠',  # 交叉状态（独热）
    'MACD距离很近', 'MACD距离近', 'MACD距离中等', 'MACD距离远',  # 距离等级（独热）
    'DIF位置附近', 'DIF位置上方中等', 'DIF位置上方远离', 'DIF位置下方接近', 'DIF位置下方远离',  # DIF位置5级（独热）
    
    # 3. 成交量特征（13个）
    'MA5成交量比例', 'MA60成交量基准',  # 相对于MA60的比例
    '量线MA5>MA60',  # 5日均量在60日均量上方
    'MA5上方占比',  # 60天中MA5>MA60的占比（持续放量程度）
    '平均缩量幅度',  # MA5<MA60时的平均缩量幅度
    '成交量上穿率',  # 原有特征
    '成交量相对强度',  # 当前成交量 / MA60成交量
    'MA60量线向上', 'MA60量线横盘', 'MA60量线向下',  # 60日均量线趋势（独热编码）
    '成交量强势股', '成交量弱势股', '成交量震荡'  # 成交量趋势类型（独热编码）
]

# 特征总数
NUM_FEATURES = len(FEATURE_NAMES)  # 43个


# ============================================================================
# 特征提取函数
# ============================================================================
def extract_features_sequence_from_kline_data(kline_data: list, period: str = 'day'):
    """
    从120根K线数据中提取60根K线的特征序列（用于LSTM模型）
    
    参数:
        kline_data: 120根K线数据列表（前60根历史 + 后60根显示）
        period: K线周期
    
    返回:
        np.array: 特征序列 (60, 43) - 60根K线，每根43个特征
    
    说明:
        - 前60根：用于计算后60根的60日均线
        - 后60根：提取每一根K线的43个特征，形成时序数据
        - 返回形状: (60, 43)，适合LSTM/GRU模型输入
    """
    try:
        if len(kline_data) < 120:
            print(f"⚠️  K线数据不足120根({len(kline_data)}根)，放弃处理")
            return None
        
        # 提取价格和成交量数据
        closes = np.array([float(k.close) for k in kline_data])
        opens = np.array([float(k.open) for k in kline_data])
        highs = np.array([float(k.high) for k in kline_data])
        lows = np.array([float(k.low) for k in kline_data])
        volumes = np.array([float(k.volume) for k in kline_data])
        
        # === 计算均线（使用全部120根数据）===
        ma5_prices = np.convolve(closes, np.ones(5)/5, mode='valid')  # 长度: 116
        ma25_prices = np.convolve(closes, np.ones(25)/25, mode='valid')  # 长度: 96
        ma5_volumes = np.convolve(volumes, np.ones(5)/5, mode='valid')  # 长度: 116
        ma60_volumes = np.convolve(volumes, np.ones(60)/60, mode='valid')  # 长度: 61
        
        # 计算MACD（基于全部120根数据）
        ema12 = closes.copy()
        ema26 = closes.copy()
        for i in range(1, len(closes)):
            ema12[i] = ema12[i-1] * (11/13) + closes[i] * (2/13)
            ema26[i] = ema26[i-1] * (25/27) + closes[i] * (2/27)
        
        dif = ema12 - ema26
        dea = dif.copy()
        for i in range(1, len(dif)):
            dea[i] = dea[i-1] * (8/10) + dif[i] * (2/10)
        
        # === 提取后60根K线的特征序列 ===
        features_sequence = []
        
        for i in range(60):
            # 当前K线在120根数据中的索引
            idx = 60 + i  # 从第61根到第120根
            
            close = closes[idx]
            open_price = opens[idx]
            high = highs[idx]
            low = lows[idx]
            volume = volumes[idx]
            
            # 对应的均线索引
            ma5_idx = idx - 4  # ma5从第5根开始
            ma25_idx = idx - 24  # ma25从第25根开始
            ma60_vol_idx = idx - 59  # ma60从第60根开始
            
            ma5 = ma5_prices[ma5_idx]
            ma25 = ma25_prices[ma25_idx]
            ma5_vol = ma5_volumes[ma5_idx]
            ma60_vol = ma60_volumes[ma60_vol_idx]
            
            features = []
            
            # === 1. 价格均线特征 (11个) ===
            features.append(ma5 / close if close > 0 else 1.0)  # MA5价格归一化（相对于收盘价）
            features.append(ma25 / close if close > 0 else 1.0)  # MA25价格归一化（相对于收盘价）
            features.append(1.0 if close > ma5 else 0.0)
            features.append(1.0 if close > ma25 else 0.0)
            features.append(1.0 if ma5 > ma25 else 0.0)
            
            # MA25趋势方向（基于当前位置往前的MA25斜率）
            if ma25_idx >= 10:
                ma25_slope = (ma25_prices[ma25_idx] - ma25_prices[ma25_idx-10]) / 10
                ma25_trend = '向上' if ma25_slope > 0.001 else ('向下' if ma25_slope < -0.001 else '横盘')
            else:
                ma25_trend = '横盘'
            features.append(1.0 if ma25_trend == '向上' else 0.0)
            features.append(1.0 if ma25_trend == '横盘' else 0.0)
            features.append(1.0 if ma25_trend == '向下' else 0.0)
            
            features.append(1.0 if close > open_price else 0.0)  # K线收阳
            features.append((close - ma25) / ma25 if ma25 > 0 else 0)  # 价格相对MA25强度
            features.append(1.0 if low > ma25 else 0.0)  # K线在MA25上方
            
            # === 2. MACD特征 (19个) ===
            dif_val = dif[idx]
            dea_val = dea[idx]
            macd_val = (dif_val - dea_val) * 2
            
            features.append(dif_val / close if close > 0 else 0)  # DIF归一化（相对于收盘价）
            features.append(dea_val / close if close > 0 else 0)  # DEA归一化（相对于收盘价）
            features.append(macd_val / close if close > 0 else 0)  # MACD柱归一化（相对于收盘价）
            features.append(abs(dif_val - dea_val) / close if close > 0 else 0)  # DIF-DEA距离归一化
            
            # MACD零轴位置
            features.append(1.0 if dif_val > 0.5 else 0.0)
            features.append(1.0 if dif_val < -0.5 else 0.0)
            features.append(1.0 if -0.5 <= dif_val <= 0.5 else 0.0)
            
            # MACD交叉状态
            if idx > 0:
                is_golden = dif_val > dea_val and dif[idx-1] <= dea[idx-1]
                is_dead = dif_val < dea_val and dif[idx-1] >= dea[idx-1]
                is_tangle = abs(dif_val - dea_val) < 0.15
            else:
                is_golden = is_dead = is_tangle = False
            features.append(1.0 if is_golden else 0.0)
            features.append(1.0 if is_dead else 0.0)
            features.append(1.0 if is_tangle else 0.0)
            
            # MACD距离等级
            dist = abs(dif_val - dea_val)
            features.append(1.0 if dist < 0.1 else 0.0)
            features.append(1.0 if 0.1 <= dist < 0.3 else 0.0)
            features.append(1.0 if 0.3 <= dist < 1.0 else 0.0)
            features.append(1.0 if dist >= 1.0 else 0.0)
            
            # DIF位置分级
            features.append(1.0 if -0.5 <= dif_val <= 0.5 else 0.0)
            features.append(1.0 if 0.5 < dif_val <= 3.0 else 0.0)
            features.append(1.0 if dif_val > 3.0 else 0.0)
            features.append(1.0 if -1.5 <= dif_val < -0.5 else 0.0)
            features.append(1.0 if dif_val < -1.5 else 0.0)
            
            # === 3. 成交量特征 (13个) ===
            features.append(ma5_vol / ma60_vol if ma60_vol > 0 else 1.0)  # MA5量归一化（相对于MA60）
            features.append(1.0)  # MA60量归一化（基准为1.0）
            features.append(1.0 if ma5_vol > ma60_vol else 0.0)
            
            # MA5在MA60上方占比（基于后60根K线的统计）
            # 计算从第61根K线到当前K线的MA5和MA60比较
            lookback = i + 1  # 当前是后60根中的第i+1根
            
            # 确定在ma5_volumes和ma60_volumes中的起始和结束索引
            # 后60根K线对应的ma5索引范围：56-115（对应K线60-119）
            # 后60根K线对应的ma60索引范围：1-60（对应K线60-119）
            
            # 对于当前K线idx，它是后60根中的第i+1根
            # 从后60根的第一根到当前根
            start_idx_in_60 = 0  # 后60根的起始位置
            end_idx_in_60 = i  # 当前位置（0-based）
            
            # 转换为ma5和ma60数组的索引
            ma5_start = 56  # 对应K线索引60的ma5索引
            ma5_end = ma5_idx  # 当前K线的ma5索引
            
            ma60_start = 1  # 对应K线索引60的ma60索引
            ma60_end = ma60_vol_idx  # 当前K线的ma60索引
            
            if ma5_end >= ma5_start and ma60_end >= ma60_start:
                ma5_window = ma5_volumes[ma5_start:ma5_end+1]
                ma60_window = ma60_volumes[ma60_start:ma60_end+1]
                
                if len(ma5_window) > 0 and len(ma60_window) > 0:
                    min_len = min(len(ma5_window), len(ma60_window))
                    above_count = np.sum(ma5_window[-min_len:] > ma60_window[-min_len:])
                    ma5_above_ratio = above_count / min_len
                else:
                    ma5_above_ratio = 0.5
            else:
                ma5_above_ratio = 0.5
            features.append(ma5_above_ratio)
            
            # 平均缩量幅度
            if ma5_end >= ma5_start and ma60_end >= ma60_start:
                ma5_window = ma5_volumes[ma5_start:ma5_end+1]
                ma60_window = ma60_volumes[ma60_start:ma60_end+1]
                
                if len(ma5_window) > 0 and len(ma60_window) > 0:
                    min_len = min(len(ma5_window), len(ma60_window))
                    below_mask = ma5_window[-min_len:] < ma60_window[-min_len:]
                    if np.any(below_mask):
                        shrink_ratios = (ma60_window[-min_len:][below_mask] - ma5_window[-min_len:][below_mask]) / ma60_window[-min_len:][below_mask]
                        avg_shrink = np.mean(shrink_ratios)
                    else:
                        avg_shrink = 0
                else:
                    avg_shrink = 0
            else:
                avg_shrink = 0
            features.append(avg_shrink)
            
            features.append(ma5_above_ratio)  # 成交量上穿率（复用）
            features.append(volume / ma60_vol if ma60_vol > 0 else 1.0)  # 成交量相对强度
            
            # MA60量线趋势
            if ma60_vol_idx >= 10:
                ma60_slope = (ma60_volumes[ma60_vol_idx] - ma60_volumes[ma60_vol_idx-10]) / 10
                ma60_trend = '向上' if ma60_slope > 0.003 else ('向下' if ma60_slope < -0.003 else '横盘')
            else:
                ma60_trend = '横盘'
            features.append(1.0 if ma60_trend == '向上' else 0.0)
            features.append(1.0 if ma60_trend == '横盘' else 0.0)
            features.append(1.0 if ma60_trend == '向下' else 0.0)
            
            # 成交量趋势类型
            is_strong = ma5_above_ratio >= 0.7
            is_weak = ma5_above_ratio < 0.5 and avg_shrink > 0.3
            features.append(1.0 if is_strong else 0.0)
            features.append(1.0 if is_weak else 0.0)
            features.append(1.0 if not is_strong and not is_weak else 0.0)
            
            # 验证特征数量
            if len(features) != NUM_FEATURES:
                print(f"⚠️  警告: 第{i+1}根K线特征数量不匹配! 期望{NUM_FEATURES}个，实际{len(features)}个")
            
            features_sequence.append(features)
        
        return np.array(features_sequence, dtype=np.float32)  # 形状: (60, 43)
    
    except Exception as e:
        print(f"特征序列提取失败: {e}")
        import traceback
        traceback.print_exc()
        return None



def extract_features_sequence_from_image(image_path, analyzer):
    """
    从股票图片中提取60根K线的特征序列（用于LSTM模型）
    
    参数:
        image_path: 图片路径
        analyzer: StockImageAnalyzer实例
    
    返回:
        np.array: 特征序列 (60, 43)，数据不足120根时返回None
        
    数据要求:
        - 必须有120根K线（前60根历史用于计算均线 + 后60根用于特征提取）
        - 优先从数据库读取
        - 数据库不足120根则直接放弃（不降级到图片识别）
    """
    # 智能获取训练数据（要求至少120根K线）
    data = analyzer.get_training_data_smart(image_path, min_klines=120)
    
    if data is None or not data.get('kline_data'):
        # 数据不足120根，直接放弃
        return None
    
    # 从120根K线数据提取60根K线的特征序列
    return extract_features_sequence_from_kline_data(data['kline_data'])


# ============================================================================
# 特征分组信息（用于可解释性分析）
# ============================================================================
def get_feature_groups():
    """
    获取特征分组信息
    
    返回:
        dict: 特征分组字典
    """
    return {
        'price': {
            'name': '价格均线组',
            'features': [
                'MA5价格', 'MA25价格', '价格>MA5', '价格>MA25',
                'MA5>MA25', 'MA25趋势向上', 'MA25趋势横盘', 'MA25趋势向下',
                'K线收阳', '价格相对MA25强度', 'K线在MA25上方'
            ],
            'indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        },
        'macd': {
            'name': 'MACD指标组',
            'features': [
                'DIF值', 'DEA值',
                'MACD柱状图值', 'DIF-DEA距离',
                'MACD零轴上方', 'MACD零轴下方', 'MACD零轴附近',
                'MACD金叉', 'MACD死叉', 'MACD纠缠',
                'MACD距离很近', 'MACD距离近', 'MACD距离中等', 'MACD距离远',
                'DIF位置附近', 'DIF位置上方中等', 'DIF位置上方远离', 'DIF位置下方接近', 'DIF位置下方远离'
            ],
            'indices': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]  # 19个特征，索引11-29
        },
        'volume': {
            'name': '成交量指标组',
            'features': [
                'MA5成交量', 'MA60成交量', '量线MA5>MA60',
                'MA5上方占比', '平均缩量幅度', '成交量上穿率', '成交量相对强度',
                'MA60量线向上', 'MA60量线横盘', 'MA60量线向下',
                '成交量强势股', '成交量弱势股', '成交量震荡'
            ],
            'indices': [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]  # 13个特征，索引30-42
        }
    }


# ============================================================================
# 特征统计信息
# ============================================================================
def print_feature_info():
    """打印特征信息"""
    print("=" * 70)
    print("AI训练特征信息")
    print("=" * 70)
    print(f"\n特征总数: {NUM_FEATURES}个")
    print("\n特征分组:")
    
    groups = get_feature_groups()
    for group_key, group_info in groups.items():
        print(f"\n【{group_info['name']}】 {len(group_info['features'])}个")
        for i, name in enumerate(group_info['features'], 1):
            idx = group_info['indices'][i-1]
            print(f"  {idx+1:2d}. {name}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    # 测试：打印特征信息
    print_feature_info()
