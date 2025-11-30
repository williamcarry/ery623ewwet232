"""
AI训练特征提取器 - 优化版本（56特征）

作用：使用连续归一化值替代独热编码，让AI自己学习阈值
优势：适应不同价位股票，减少主观阈值设定

特征总数：56个（基础46 + 新增10个K线与支撑阻力特征）
版本：v3.3 - 新增10个K线形态和支撑阻力特征，为所有特征添加编码标识

文档参考：
  - FEATURE_ANALYSIS.md：问题0（纠缠状态核心）、问题2（K线形态）、问题4（支撑阻力）
  - OPTIMIZATION_PLAN.md：P1优先级特征补充

============================================================================
# 特征名称定义 + 编码标识（用于可解释性分析与追踪）
============================================================================

【第1组】价格均线特征（14个）- 使用连续值 + K线方向强度 + 纠缠状态
┌─编码格式：F01_nn  (nn=01-14)
│
├─ F01_01: 'MA5价格归一化'               # MA5 / Close
├─ F01_02: 'MA25价格归一化'              # MA25 / Close
├─ F01_03: '价格>MA5'                     # 布尔特征（0/1）
├─ F01_04: '价格>MA25'                    # 布尔特征（0/1）
├─ F01_05: 'MA5>MA25'                     # MA5均线在MA25均线上方（金叉状态）
├─ F01_06: 'MA25趋势斜率'                 # MA25相对斜率（25根K线），归一化
├─ F01_07: 'K线收阳'                      # K线收阳线状态（0/1）
├─ F01_08: '价格相对MA25强度'             # (Close - MA25) / MA25
├─ F01_09: 'K线方向强度'                  # (-1到+1，代表K线方向和幅度）
├─ F01_10: '红绿K线比例'                  # 最近5根中红色K线的占比
├─ F01_11: 'MA5-MA25粘合度'              # ★ P0-ENTANGLEMENT-001：|MA5-MA25|/MA25
├─ F01_12: 'MA5-MA25发散速度'            # ★ P0-ENTANGLEMENT-002：粘合度变化率
├─ F01_13: 'K线实体率'                   # ★ P1-CANDLE-001：|Close-Open|/ATR（K线力度）
└─ F01_14: 'K线影线强度'                 # ★ P1-CANDLE-002：(High-Max(Close,Open))/ATR（上影压力）

【第2组】MACD特征（10个）- 使用连续归一化值 + 纠缠状态
┌─编码格式：F02_nn  (nn=01-10)
│
├─ F02_01: 'DIF归一化'                    # DIF / Close，相对于股价的比例
├─ F02_02: 'DEA归一化'                    # DEA / Close
├─ F02_03: 'MACD柱归一化'                 # MACD / Close
├─ F02_04: 'DIF-DEA距离归一化'            # |DIF - DEA| / Close（即DIF-DEA粘合度）
├─ F02_05: 'MACD金叉'                     # 金叉信号（0/1）
├─ F02_06: 'MACD死叉'                     # 死叉信号（0/1）
├─ F02_07: 'DIF变化率'                    # (DIF_now - DIF_prev) / Close，动量
├─ F02_08: 'DEA变化率'                    # (DEA_now - DEA_prev) / Close
├─ F02_09: 'DIF-DEA发散速度'             # ★ P0-ENTANGLEMENT-003：粘合度变化率
└─ F02_10: 'MACD柱加速度'                # ★ P1-MACD-ADV：MACD柱变化率二阶导数

【第3组】成交量特征（12个）- 使用连续归一化值 + 纠缠状态
┌─编码格式：F03_nn  (nn=01-12)
│
├─ F03_01: 'MA5量归一化'                  # MA5_vol / MA60_vol
├─ F03_02: 'MA60量归一化'                 # MA60_vol基准（=1.0）
├─ F03_03: '量线MA5>MA60'                 # 5日均量在60日均量上方（0/1）
├─ F03_04: 'MA5上方占比'                  # 最近15根中MA5>MA60的占比
├─ F03_05: '平均缩量幅度'                 # MA5<MA60时的平均缩量幅度
├─ F03_06: '成交量相对强度'               # 当前成交量 / MA60成交量
├─ F03_07: 'MA60量线趋势斜率'             # MA60量线相对斜率（25根K线），归一化
├─ F03_08: '成交量波动率'                 # 近期成交量标准差 / MA60
├─ F03_09: '量线MA5-MA60粘合度'          # ★ P0-ENTANGLEMENT-004：|MA5_vol-MA60_vol|/MA60_vol
├─ F03_10: '量线MA5-MA60发散速度'        # ★ P0-ENTANGLEMENT-005：粘合度变化率
├─ F03_11: '成交量异常倍数'              # ★ P1-VOLUME-ADV-001：MA5_vol/MA60_vol非线性映射
└─ F03_12: '价量同步性'                  # ★ P1-VOLUME-ADV-002：价格变化与成交量变化的同向性

【第4组】波动率特征（4个）- 补充波动率维度
┌─编码格式：F04_nn  (nn=01-04)
│
├─ F04_01: 'ATR归一化'                    # ATR(14) / Close，平均真实波动幅度
├─ F04_02: '布林带位置'                   # (Close - 下轨) / (上轨 - 下轨)，相对位置
├─ F04_03: '布林带宽度归一化'             # (上轨 - 下轨) / 中轨，波动率水平
└─ F04_04: '波动率加速度'                 # ATR的变化率（14根）

【第5组】相对强度特征（5个）- 补充相对强度 + 多周期RSI
┌─编码格式：F05_nn  (nn=01-05)
│
├─ F05_01: 'RSI归一化'                    # RSI(14) / 100，标准EMA平滑
├─ F05_02: 'RSI快线'                      # ★ P2-RSI-MULTI-001：RSI(5) / 100（短期强度）
├─ F05_03: '价格动量20日'                 # (Close - Close_20) / Close_20，中期动量
├─ F05_04: '价格位置60日'                 # (Close - Low_60) / (High_60 - Low_60)，相对位置
└─ F05_05: '距离MA25百分比'              # ★ P2-SUPPORT-01：(Close - MA25) / MA25 * 100（支撑距离）

【第6组】强势特征（4个）- 增强趋势预测 + KDJ完善
┌─编码格式：F06_nn  (nn=01-04)
│
├─ F06_01: 'KDJ快线值'                    # 标准KDJ K线值（0-1）
├─ F06_02: 'KDJ_D线值'                   # ★ P2-KDJ-ADV-001：KDJ D线值（0-1）
├─ F06_03: '价格加速度'                   # 二阶动量（momentum_20 - momentum_10）
└─ F06_04: '连续K线强度'                  # 连续涨跌幅度和强度（-1到+1）

【第7组】量能特征（3个）- 增强量能配合度
┌─编码格式：F07_nn  (nn=01-03)
│
├─ F07_01: '成交量加速度'                 # 成交量动量的二阶导数
├─ F07_02: '价量相关性'                   # 价格上升时是否伴随放量
└─ F07_03: '成交量极值占比'               # 最近10根中成交量的极值占比

【第8组】趋势特征（2个）- 增强趋势判断
┌─编码格式：F08_nn  (nn=01-02)
│
├─ F08_01: '高低点均值偏离'               # 价格与(High+Low)/2的偏离度
└─ F08_02: '缺口强度'                     # |Close - Close_prev| / Close_prev

【第9组】支撑阻力特征（2个）- 新增（来自OPTIMIZATION_PLAN.md P2）
┌─编码格式：F09_nn  (nn=01-02)
│
├─ F09_01: '距离MA5百分比'               # ★ P2-SUPPORT-02：(Close - MA5) / MA5 * 100（短期支撑）
└─ F09_02: 'MA5支撑强度'                 # ★ P2-SUPPORT-03：当MA5被击穿时的缓冲程度（ATR）

特征总数：14 + 10 + 12 + 4 + 5 + 4 + 3 + 2 + 2 = 56个

编码规则：
  F01-F08: 原有特征组（46个）
  F09:     新增支撑阻力特征（2个）
  新增MACD和成交量高级特征（8个）

来源标识：
  ★ P0-ENTANGLEMENT：纠缠状态系统（5个特征，P0最高优先级）
  ★ P1-CANDLE：K线形态特征（2个特征，P1高优先级）
  ★ P1-MACD-ADV：MACD高级特征（1个特征）
  ★ P1-VOLUME-ADV：成交量高级特征（2个特征）
  ★ P2-RSI-MULTI：多周期RSI（1个特征）
  ★ P2-KDJ-ADV：KDJ高级特征（1个特征）
  ★ P2-SUPPORT：支撑阻力特征（3个特征）

说明：
  ✓ 第1组增加4个特征：2个纠缠状态 + 2个K线形态
  ✓ 第2组增加1个特征：MACD柱加速度
  ✓ 第3组增加2个特征：成交量异常倍数、价量同步性
  ✓ 第5组增加2个特征：RSI快线、距离MA25百分比
  ✓ 第6组增加1个特征：KDJ_D线值
  ✓ 第9组新增2个特征：距离MA5百分比、MA5支撑强度
  ✓ 纠缠状态 = 粘合度（静态距离） + 发散速度（动态变化趋势）
  ✓ 所有特征都使用连续归一化值，避免硬阈值设定
  ✓ 采用标准EMA平滑（RSI、KDJ等），确保数值稳定性

============================================================================
"""
import numpy as np
import math


# ============================================================================
# 工具函数
# ============================================================================
def safe_divide(numerator, denominator, default=0.0):
    """安全的除法操作，避免除以零"""
    return float(numerator) / float(denominator) if denominator != 0 else default


def calculate_ema(data, period):
    """
    计算指数移动平均线（EMA）
    
    参数:
        data: 价格数据数组
        period: EMA周期
    
    返回:
        EMA数组
    """
    ema = np.zeros(len(data))
    alpha = 2.0 / (period + 1)
    ema[0] = data[0]
    
    for i in range(1, len(data)):
        ema[i] = ema[i-1] * (1 - alpha) + data[i] * alpha
    
    return ema


def calculate_rsi_standard(closes, period=14):
    """
    计算标准RSI指标（使用威尔德斯平均，而不是简单平均）
    
    参数:
        closes: 收盘价数据数组
        period: RSI周期（默认14）
    
    返回:
        RSI数组
    """
    rsi = np.zeros(len(closes))
    
    if len(closes) < period + 1:
        return rsi
    
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    alpha = 1.0 / period
    
    # 初始化平均值
    avg_gain = np.sum(gains[:period]) / period
    avg_loss = np.sum(losses[:period]) / period
    
    for i in range(period, len(closes)):
        if i > period:
            avg_gain = avg_gain * (1 - alpha) + gains[i-1] * alpha
            avg_loss = avg_loss * (1 - alpha) + losses[i-1] * alpha
        else:
            avg_gain = avg_gain * (1 - alpha) + gains[i-1] * alpha
            avg_loss = avg_loss * (1 - alpha) + losses[i-1] * alpha
        
        if avg_loss == 0:
            rsi[i] = 100 if avg_gain > 0 else 50
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_kdj_standard(closes, highs, lows, period=9, k_period=3, d_period=3):
    """
    计算标准KDJ指标（随机指标）
    
    参数:
        closes: 收盘价数据数组
        highs: 最高价数据数组
        lows: 最低价数据数组
        period: KDJ周期（默认9）
        k_period: K线EMA周期（默认3）
        d_period: D线EMA周期（默认3）
    
    返回:
        (K线, D线, J线) 三个数组
    """
    k_line = np.zeros(len(closes))
    d_line = np.zeros(len(closes))
    j_line = np.zeros(len(closes))
    
    if len(closes) < period:
        return k_line, d_line, j_line
    
    # 计算RSV（未平滑的随机指标）
    rsv = np.zeros(len(closes))
    for i in range(period - 1, len(closes)):
        high_period = np.max(highs[i - period + 1:i + 1])
        low_period = np.min(lows[i - period + 1:i + 1])
        
        if high_period != low_period:
            rsv[i] = (closes[i] - low_period) / (high_period - low_period) * 100
        else:
            rsv[i] = 50
    
    # 计算K线（RSV的EMA）
    alpha_k = 2.0 / (k_period + 1)
    k_line[period - 1] = rsv[period - 1]
    for i in range(period, len(closes)):
        k_line[i] = k_line[i - 1] * (1 - alpha_k) + rsv[i] * alpha_k
    
    # 计算D线（K线的EMA）
    alpha_d = 2.0 / (d_period + 1)
    d_line[period - 1] = k_line[period - 1]
    for i in range(period, len(closes)):
        d_line[i] = d_line[i - 1] * (1 - alpha_d) + k_line[i] * alpha_d
    
    # 计算J线（3K - 2D）
    j_line = 3 * k_line - 2 * d_line
    
    return k_line, d_line, j_line


def calculate_atr(highs, lows, closes, period=14):
    """
    计算平均真实波动幅度（ATR）
    
    参数:
        highs: 最高价数据数组
        lows: 最低价数据数组
        closes: 收盘价数据数组
        period: ATR周期（默认14）
    
    返回:
        ATR数组
    """
    atr = np.zeros(len(closes))
    
    if len(closes) < period:
        return atr
    
    true_ranges = []
    for i in range(len(closes)):
        if i == 0:
            tr = highs[i] - lows[i]
        else:
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
        true_ranges.append(tr)
    
    true_ranges = np.array(true_ranges)
    atr[0:period] = np.mean(true_ranges[0:period])
    
    for i in range(period, len(closes)):
        atr[i] = (atr[i-1] * (period - 1) + true_ranges[i]) / period
    
    return atr


def calculate_bollinger_bands(data, period=20, num_std=2):
    """
    计算布林带（上轨、中轨、下轨）
    
    参数:
        data: 价格数据数组
        period: 布林带周期（默认20）
        num_std: 标准差倍数（默认2）
    
    返回:
        (upper_band, middle_band, lower_band) 元组
    """
    upper_band = np.zeros(len(data))
    middle_band = np.zeros(len(data))
    lower_band = np.zeros(len(data))
    
    for i in range(len(data)):
        if i < period - 1:
            start_idx = 0
        else:
            start_idx = i - period + 1
        
        window = data[start_idx:i+1]
        middle = np.mean(window)
        std = np.std(window)
        
        middle_band[i] = middle
        upper_band[i] = middle + num_std * std
        lower_band[i] = middle - num_std * std
    
    return upper_band, middle_band, lower_band


def _get_ma_windows(ma_prices, ma_volumes, idx, ma_period, ma_vol_period, window_size=15):
    """
    获取均线滑动窗口（优化：复用逻辑，避免重复）
    
    参数:
        ma_prices: 均线价格数组
        ma_volumes: 均线成交量数组
        idx: 当前K线在原始数据中的索引
        ma_period: 价格均线周期
        ma_vol_period: 成交量均线周期
        window_size: 窗口大小（默认15）
    
    返回:
        (ma_idx, ma_vol_idx, ma_recent, ma_vol_recent) 元组
    """
    ma_idx = idx - (ma_period - 1)
    ma_vol_idx = idx - (ma_vol_period - 1)
    
    # 提取最近window_size个值
    ma_start = max(0, ma_idx - (window_size - 1))
    ma_vol_start = max(0, ma_vol_idx - (window_size - 1))
    
    ma_recent = ma_prices[ma_start:ma_idx + 1]
    ma_vol_recent = ma_volumes[ma_vol_start:ma_vol_idx + 1]
    
    return ma_idx, ma_vol_idx, ma_recent, ma_vol_recent


# ============================================================================
# 特征名称定义 + 编码标识（用于可解释性分析）
# ============================================================================
FEATURE_NAMES = [
    # 【第1组】价格均线特征（14个）
    'F01_01: MA5价格归一化',
    'F01_02: MA25价格归一化',
    'F01_03: 价格>MA5',
    'F01_04: 价格>MA25',
    'F01_05: MA5>MA25',
    'F01_06: MA25趋势斜率',
    'F01_07: K线收阳',
    'F01_08: 价格相对MA25强度',
    'F01_09: K线方向强度',
    'F01_10: 红绿K线比例',
    'F01_11: MA5-MA25粘合度 [P0-ENTANGLEMENT-001]',
    'F01_12: MA5-MA25发散速度 [P0-ENTANGLEMENT-002]',
    'F01_13: K线实体率 [P1-CANDLE-001]',
    'F01_14: K线影线强度 [P1-CANDLE-002]',

    # 【第2组】MACD特征（10个）
    'F02_01: DIF归一化',
    'F02_02: DEA归一化',
    'F02_03: MACD柱归一化',
    'F02_04: DIF-DEA距离归一化',
    'F02_05: MACD金叉',
    'F02_06: MACD死叉',
    'F02_07: DIF变化率',
    'F02_08: DEA变化率',
    'F02_09: DIF-DEA发散速度 [P0-ENTANGLEMENT-003]',
    'F02_10: MACD柱加速度 [P1-MACD-ADV]',

    # 【第3组】成交量特征（12个）
    'F03_01: MA5量归一化',
    'F03_02: MA60量归一化',
    'F03_03: 量线MA5>MA60',
    'F03_04: MA5上方占比',
    'F03_05: 平均缩量幅度',
    'F03_06: 成交量相对强度',
    'F03_07: MA60量线趋势斜率',
    'F03_08: 成交量波动率',
    'F03_09: 量线MA5-MA60粘合度 [P0-ENTANGLEMENT-004]',
    'F03_10: 量线MA5-MA60发散速度 [P0-ENTANGLEMENT-005]',
    'F03_11: 成交量异常倍数 [P1-VOLUME-ADV-001]',
    'F03_12: 价量同步性 [P1-VOLUME-ADV-002]',

    # 【第4组】波动率特征（4个）
    'F04_01: ATR归一化',
    'F04_02: 布林带位置',
    'F04_03: 布林带宽度归一化',
    'F04_04: 波动率加速度',

    # 【第5组】相对强度特征（5个）
    'F05_01: RSI归一化',
    'F05_02: RSI快线 [P2-RSI-MULTI-001]',
    'F05_03: 价格动量20日',
    'F05_04: 价格位置60日',
    'F05_05: 距离MA25百分比 [P2-SUPPORT-01]',

    # 【第6组】强势特征（4个）
    'F06_01: KDJ快线值',
    'F06_02: KDJ_D线值 [P2-KDJ-ADV-001]',
    'F06_03: 价格加速度',
    'F06_04: 连续K线强度',

    # 【第7组】量能特征（3个）
    'F07_01: 成交量加速度',
    'F07_02: 价量相关性',
    'F07_03: 成交量极值占比',

    # 【第8组】趋势特征（2个）
    'F08_01: 高低点均值偏离',
    'F08_02: 缺口强度',

    # 【第9组】支撑阻力特征（2个）
    'F09_01: 距离MA5百分比 [P2-SUPPORT-02]',
    'F09_02: MA5支撑强度 [P2-SUPPORT-03]',
]

NUM_FEATURES = len(FEATURE_NAMES)  # 56个 = 14+10+12+4+5+4+3+2+2


# ============================================================================
# 特征提取函数（时序版本）
# ============================================================================
def extract_features_sequence_from_kline_data(kline_data: list, period: str = 'day'):
    """
    从120根K线数据中提取60根K线的特征序列（用于LSTM模型）
    
    参数:
        kline_data: 120根K线数据列表（前60根历史 + 后60根显示）
        period: K线周期
    
    返回:
        np.array: 特征序列 (60, 42) - 60根K线，每根42个特征
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
        ma5_prices = np.convolve(closes, np.ones(5)/5, mode='valid')
        ma25_prices = np.convolve(closes, np.ones(25)/25, mode='valid')
        ma5_volumes = np.convolve(volumes, np.ones(5)/5, mode='valid')
        ma60_volumes = np.convolve(volumes, np.ones(60)/60, mode='valid')
        
        # === 计算高级指标 ===
        ema12 = calculate_ema(closes, 12)
        ema26 = calculate_ema(closes, 26)
        dif = ema12 - ema26
        dea = calculate_ema(dif, 9)

        rsi = calculate_rsi_standard(closes, 14)
        rsi5 = calculate_rsi_standard(closes, 5)  # 新增：RSI快线（5周期）
        kdj_k, kdj_d, kdj_j = calculate_kdj_standard(closes, highs, lows, 9, 3, 3)
        atr = calculate_atr(highs, lows, closes, 14)
        upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(closes, 20, 2)

        # 计算MACD柱值数组（用于计算加速度）
        macd_histogram = (dif - dea) * 2
        
        # === 提取后60根K线的特征序列 ===
        features_sequence = []
        
        # 用于计算连续K线数
        consecutive_up = 0
        consecutive_down = 0
        
        # 用于计算成交量加速度
        prev_vol_momentum = 0
        
        for i in range(60):
            idx = 60 + i  # 从第61根到第120根
            
            close = closes[idx]
            open_price = opens[idx]
            high = highs[idx]
            low = lows[idx]
            volume = volumes[idx]
            prev_close = closes[idx-1] if idx > 0 else close
            
            # 对应的均线索引
            ma5_idx = idx - 4
            ma25_idx = idx - 24
            ma60_vol_idx = idx - 59
            
            ma5 = ma5_prices[ma5_idx]
            ma25 = ma25_prices[ma25_idx]
            ma5_vol = ma5_volumes[ma5_idx]
            ma60_vol = ma60_volumes[ma60_vol_idx]
            
            features = []
            
            # === 1. 价格均线特征 (10个) - 优化版本 ===
            features.append(safe_divide(ma5, close, 1.0))  # MA5价格归一化
            features.append(safe_divide(ma25, close, 1.0))  # MA25价格归一化
            features.append(1.0 if close > ma5 else 0.0)  # 价格>MA5
            features.append(1.0 if close > ma25 else 0.0)  # 价格>MA25
            features.append(1.0 if ma5 > ma25 else 0.0)  # MA5>MA25
            
            # MA25趋势斜率（优化：使用更长周期25根）
            if ma25_idx >= 25:
                ma25_slope = (ma25_prices[ma25_idx] - ma25_prices[ma25_idx-25]) / 25
                ma25_slope_normalized = safe_divide(ma25_slope, ma25, 0)
            else:
                ma25_slope_normalized = 0
            features.append(ma25_slope_normalized)
            
            features.append(1.0 if close > open_price else 0.0)  # K线收阳
            features.append(safe_divide(close - ma25, ma25, 0))  # 价格相对MA25强度
            
            # K线方向强度（-1到+1）
            k_line_strength = safe_divide(close - open_price, abs(close - open_price) + 0.0001, 0)
            features.append(k_line_strength)
            
            # 红绿 K线比例（最近5根）
            if i >= 5:
                red_count = sum([1 for j in range(i-4, i+1) if closes[60+j] > opens[60+j]])
                red_ratio = red_count / 5.0
            else:
                red_ratio = 0.5
            features.append(red_ratio)
                        
            # === 纠缠状态特征1：MA5-MA25粘合度 ===
            ma_cohesion = safe_divide(abs(ma5 - ma25), ma25, 0)
            features.append(ma_cohesion)  # MA5-MA25粘合度
                        
            # === 纠缠状态特征2：MA5-MA25发散速度 ===
            if i >= 5 and ma5_idx >= 5:
                # 计算5根K线前的粘合度
                prev_ma5 = ma5_prices[ma5_idx - 5]
                prev_ma25 = ma25_prices[ma25_idx - 5] if ma25_idx >= 5 else ma25
                prev_cohesion = safe_divide(abs(prev_ma5 - prev_ma25), prev_ma25, 0)
                # 发散速度 = 粘合度变化率
                ma_divergence_speed = safe_divide(ma_cohesion - prev_cohesion, prev_cohesion + 0.0001, 0)
            else:
                ma_divergence_speed = 0
            features.append(ma_divergence_speed)  # MA5-MA25发散速度

            # === F01_13: K线实体率 [P1-CANDLE-001] ===
            # 定义：K线实体相对于ATR的比例，反映K线力度
            atr_val = atr[idx]
            k_line_body = abs(close - open_price)
            k_line_body_ratio = safe_divide(k_line_body, atr_val, 0)
            k_line_body_ratio = min(k_line_body_ratio, 2.0)  # 上限归一化
            features.append(k_line_body_ratio)

            # === F01_14: K线影线强度 [P1-CANDLE-002] ===
            # 定义：上影线（压力）相对于ATR的比例
            k_line_upper_shadow = high - max(close, open_price)
            k_line_upper_shadow_ratio = safe_divide(k_line_upper_shadow, atr_val, 0)
            k_line_upper_shadow_ratio = min(k_line_upper_shadow_ratio, 2.0)  # 上限归一化
            features.append(k_line_upper_shadow_ratio)
            
            # === 2. MACD特征 (8个) ===
            dif_val = dif[idx]
            dea_val = dea[idx]
            macd_val = (dif_val - dea_val) * 2
            
            features.append(safe_divide(dif_val, close, 0))  # DIF归一化
            features.append(safe_divide(dea_val, close, 0))  # DEA归一化
            features.append(safe_divide(macd_val, close, 0))  # MACD柱归一化
            features.append(safe_divide(abs(dif_val - dea_val), close, 0))  # DIF-DEA距离
            
            # MACD交叉信号
            if idx > 0:
                is_golden = dif_val > dea_val and dif[idx-1] <= dea[idx-1]
                is_dead = dif_val < dea_val and dif[idx-1] >= dea[idx-1]
            else:
                is_golden = is_dead = False
            features.append(1.0 if is_golden else 0.0)  # MACD金叉
            features.append(1.0 if is_dead else 0.0)  # MACD死叉
            
            # MACD变化率
            if idx > 0:
                dif_change = safe_divide(dif_val - dif[idx-1], close, 0)
                dea_change = safe_divide(dea_val - dea[idx-1], close, 0)
            else:
                dif_change = dea_change = 0
            features.append(dif_change)  # DIF变化率
            features.append(dea_change)  # DEA变化率
            
            # === 纠缠状态特征3：DIF-DEA发散速度 ===
            dif_dea_cohesion = safe_divide(abs(dif_val - dea_val), close, 0)  # 当前粘合度
            if i >= 5 and idx >= 5:
                # 计算5根K线前的粘合度
                prev_dif = dif[idx - 5]
                prev_dea = dea[idx - 5]
                prev_close = closes[idx - 5]
                prev_dif_dea_cohesion = safe_divide(abs(prev_dif - prev_dea), prev_close, 0)
                # 发散速度 = 粘合度变化率
                dif_dea_divergence_speed = safe_divide(
                    dif_dea_cohesion - prev_dif_dea_cohesion,
                    prev_dif_dea_cohesion + 0.0001,
                    0
                )
            else:
                dif_dea_divergence_speed = 0
            features.append(dif_dea_divergence_speed)  # DIF-DEA发散速度

            # === F02_10: MACD柱加速度 [P1-MACD-ADV] ===
            # 定义：MACD柱值的二阶动量（变化率的变化率）
            if idx > 1:
                macd_now = macd_histogram[idx]
                macd_prev = macd_histogram[idx - 1]
                macd_prev_prev = macd_histogram[idx - 2]
                macd_momentum_now = safe_divide(macd_now - macd_prev, abs(macd_prev) + 0.0001, 0)
                macd_momentum_prev = safe_divide(macd_prev - macd_prev_prev, abs(macd_prev_prev) + 0.0001, 0)
                macd_acceleration = macd_momentum_now - macd_momentum_prev
            else:
                macd_acceleration = 0
            features.append(macd_acceleration)

            # === 3. 成交量特征 (12个) - 优化版本 + 新增高级特征 ===
            features.append(safe_divide(ma5_vol, ma60_vol, 1.0))  # MA5量归一化
            features.append(1.0)  # MA60量归一化（基准为1.0）
            features.append(1.0 if ma5_vol > ma60_vol else 0.0)  # 量线MA5>MA60
            
            # 提取均线窗口（优化：复用逻辑）
            if idx >= 60:
                _, _, ma5_recent, ma60_vol_recent = _get_ma_windows(
                    ma5_prices, ma60_volumes, idx, 5, 60, window_size=15
                )
                min_len = min(len(ma5_recent), len(ma60_vol_recent))
            else:
                min_len = 0
            
            # MA5上方占比
            if min_len > 0:
                above_count = np.sum(ma5_recent[-min_len:] > ma60_vol_recent[-min_len:])
                ma5_above_ratio = above_count / min_len
            else:
                ma5_above_ratio = 0.5
            features.append(ma5_above_ratio)
            
            # 平均缩量幅度
            if min_len > 0:
                below_mask = ma5_recent[-min_len:] < ma60_vol_recent[-min_len:]
                if np.any(below_mask):
                    shrink_ratios = safe_divide(
                        ma60_vol_recent[-min_len:][below_mask] - ma5_recent[-min_len:][below_mask],
                        ma60_vol_recent[-min_len:][below_mask],
                        0
                    )
                    avg_shrink = np.mean(shrink_ratios) if isinstance(shrink_ratios, np.ndarray) else shrink_ratios
                else:
                    avg_shrink = 0
            else:
                avg_shrink = 0
            features.append(avg_shrink)
            
            features.append(safe_divide(volume, ma60_vol, 1.0))  # 成交量相对强度
            
            # MA60量线趋势斜率
            if ma60_vol_idx >= 25:
                ma60_slope = (ma60_volumes[ma60_vol_idx] - ma60_volumes[ma60_vol_idx-25]) / 25
                ma60_slope_normalized = safe_divide(ma60_slope, ma60_vol, 0)
            else:
                ma60_slope_normalized = 0
            features.append(ma60_slope_normalized)
            
            # 成交量波动率
            if min_len > 1:
                vol_std = np.std(ma5_recent)
                vol_volatility = safe_divide(vol_std, ma60_vol, 0)
            else:
                vol_volatility = 0
            features.append(vol_volatility)
            
            # === 纠缠状态特征4：量线MA5-MA60粘合度 ===
            vol_cohesion = safe_divide(abs(ma5_vol - ma60_vol), ma60_vol, 0)
            features.append(vol_cohesion)  # 量线MA5-MA60粘合度
            
            # === 纠缠状态特征5：量线MA5-MA60发散速度 ===
            if i >= 5 and ma5_idx >= 5 and ma60_vol_idx >= 5:
                # 计算5根K线前的粘合度
                prev_ma5_vol = ma5_volumes[ma5_idx - 5]
                prev_ma60_vol = ma60_volumes[ma60_vol_idx - 5]
                prev_vol_cohesion = safe_divide(abs(prev_ma5_vol - prev_ma60_vol), prev_ma60_vol, 0)
                # 发散速度 = 粘合度变化率
                vol_divergence_speed = safe_divide(
                    vol_cohesion - prev_vol_cohesion,
                    prev_vol_cohesion + 0.0001,
                    0
                )
            else:
                vol_divergence_speed = 0
            features.append(vol_divergence_speed)  # 量线MA5-MA60发散速度

            # === F03_11: 成交量异常倍数 [P1-VOLUME-ADV-001] ===
            # 定义：MA5量线与MA60量线的比值，用log映射处理极端值
            vol_ratio = safe_divide(ma5_vol, ma60_vol, 1.0)
            # 使用log变换：log(x) 将比值映射到更线性的空间
            # 1.0 -> log(1.0) = 0
            # 2.0 -> log(2.0) ≈ 0.693
            # 0.5 -> log(0.5) ≈ -0.693
            vol_anomaly_ratio = safe_divide(
                math.log(max(0.1, vol_ratio) + 0.0001),
                math.log(3.0),  # 归一化：log(3.0) ≈ 1.1
                0
            )
            vol_anomaly_ratio = max(-1.0, min(1.0, vol_anomaly_ratio))
            features.append(vol_anomaly_ratio)

            # === F03_12: 价量同步性 [P1-VOLUME-ADV-002] ===
            # 定义：价格变化与成交量变化的同向性判断
            if i >= 5 and ma5_idx >= 5:
                # 计算5根K线的价格变化
                price_change_5 = safe_divide(close - closes[idx - 5], closes[idx - 5], 0)
                # 计算5根K线的成交量变化
                vol_change_5 = safe_divide(ma5_vol - ma5_volumes[ma5_idx - 5], ma5_volumes[max(0, ma5_idx - 5)] + 0.0001, 0)
                # 同步性：正相关（价涨量增或价跌量缩） = 正值
                #         负相关（价涨量缩或价跌量增） = 负值
                price_vol_sync = (price_change_5 * vol_change_5) / (abs(price_change_5) + abs(vol_change_5) + 0.0001)
            else:
                price_vol_sync = 0
            features.append(price_vol_sync)

            # === 4. 波动率特征 (4个) - 增强版本 ===
            atr_val = atr[idx]
            features.append(safe_divide(atr_val, close, 0))  # ATR归一化
            
            # 布林带位置
            band_width = upper_bb[idx] - lower_bb[idx]
            if band_width > 0:
                bollinger_position = safe_divide(close - lower_bb[idx], band_width, 0.5)
                bollinger_position = max(0, min(1, bollinger_position))
            else:
                bollinger_position = 0.5
            features.append(bollinger_position)
            
            # 布林带宽度归一化
            features.append(safe_divide(band_width, middle_bb[idx], 0))
            
            # 波动率加速度（ATR的变化率）
            if idx >= 14:
                atr_momentum = safe_divide(atr[idx] - atr[idx-14], atr[idx-14], 0)
            else:
                atr_momentum = 0
            features.append(atr_momentum)
            
            # === 5. 相对强度特征 (3个) ===
            rsi_val = rsi[idx]
            features.append(rsi_val / 100.0)  # RSI归一化（标准EMA平滑）
            
            # 价格动量20日
            if idx >= 20:
                close_20_ago = closes[idx-20]
                momentum_20 = safe_divide(close - close_20_ago, close_20_ago, 0)
            else:
                momentum_20 = 0
            features.append(momentum_20)
            
            # 价格位置60日
            if idx >= 60:
                price_window = closes[idx-59:idx+1]
                high_60 = np.max(price_window)
                low_60 = np.min(price_window)
                if high_60 > low_60:
                    price_position = safe_divide(close - low_60, high_60 - low_60, 0.5)
                else:
                    price_position = 0.5
            else:
                price_window = closes[0:idx+1]
                if len(price_window) > 1:
                    high_60 = np.max(price_window)
                    low_60 = np.min(price_window)
                    if high_60 > low_60:
                        price_position = safe_divide(close - low_60, high_60 - low_60, 0.5)
                    else:
                        price_position = 0.5
                else:
                    price_position = 0.5
            features.append(price_position)

            # === F05_02: RSI快线 [P2-RSI-MULTI-001] ===
            # 定义：短期RSI（5周期），反映短期超买超卖状态
            rsi5_val = rsi5[idx]
            features.append(rsi5_val / 100.0)  # RSI(5) 归一化到 [0, 1]

            # === F05_05: 距离MA25百分比 [P2-SUPPORT-01] ===
            # 定义：当前价格与MA25的距离百分比，正值表示超买，负值表示超卖
            distance_ma25_percent = safe_divide(close - ma25, ma25, 0) * 100
            # 归一化到 [-1, 1] 的范围（假设±5%为合理范围）
            distance_ma25_percent_normalized = max(-1.0, min(1.0, distance_ma25_percent / 5.0))
            features.append(distance_ma25_percent_normalized)

            # === 6. 新增强势特征 (4个) - 增加KDJ_D线 ===
            # KDJ快线值（标准计算，已平滑）
            kdj_k_val = kdj_k[idx]
            kdj_k_normalized = safe_divide(kdj_k_val, 100.0, 0.5)
            kdj_k_normalized = max(0, min(1, kdj_k_normalized))
            features.append(kdj_k_normalized)

            # === F06_02: KDJ_D线值 [P2-KDJ-ADV-001] ===
            # 定义：KDJ D线（K线的平滑线），反映较平稳的随机指标
            kdj_d_val = kdj_d[idx]
            kdj_d_normalized = safe_divide(kdj_d_val, 100.0, 0.5)
            kdj_d_normalized = max(0, min(1, kdj_d_normalized))
            features.append(kdj_d_normalized)

            # 价格加速度（二阶动量）
            if idx >= 20:
                momentum_10 = safe_divide(close - closes[idx-10], closes[idx-10], 0)
                price_acceleration = momentum_20 - momentum_10
            else:
                price_acceleration = 0
            features.append(price_acceleration)
            
            # 连续K线强度（计算连续涨跌的幅度）
            if idx > 0:
                if close > prev_close:
                    consecutive_up += 1
                    consecutive_down = 0
                else:
                    consecutive_down += 1
                    consecutive_up = 0
                
                consecutive_strength = (consecutive_up - consecutive_down) / 10.0
                consecutive_strength = max(-1, min(1, consecutive_strength))
            else:
                consecutive_strength = 0
            features.append(consecutive_strength)
            
            # === 7. 新增量能特征 (3个) ===
            # 成交量加速度
            if idx > 0:
                prev_ma5_vol = ma5_volumes[ma5_idx-1] if ma5_idx > 0 else ma5_vol
                vol_momentum = safe_divide(ma5_vol - prev_ma5_vol, prev_ma5_vol, 0)
                vol_acceleration = vol_momentum - prev_vol_momentum
                prev_vol_momentum = vol_momentum
            else:
                vol_acceleration = 0
            features.append(vol_acceleration)
            
            # 价量相关性（价格上升时是否伴随放量）
            if idx >= 5:
                price_change = safe_divide(close - closes[idx-5], closes[idx-5], 0)
                vol_change = safe_divide(ma5_vol - ma5_volumes[ma5_idx-5], ma5_volumes[max(0, ma5_idx-5)], 0)
                price_vol_correlation = (price_change * vol_change) / (abs(price_change) + abs(vol_change) + 0.0001)
            else:
                price_vol_correlation = 0
            features.append(price_vol_correlation)
            
            # 成交量极值占比（最近10个中是否有极值）
            if i >= 10:
                vol_window = volumes[60+i-9:60+i+1]
                vol_max = np.max(vol_window)
                vol_min = np.min(vol_window)
                if vol_max > vol_min:
                    vol_extreme_ratio = max(
                        safe_divide(volume - vol_min, vol_max - vol_min, 0),
                        safe_divide(vol_max - volume, vol_max - vol_min, 0)
                    )
                else:
                    vol_extreme_ratio = 0
            else:
                vol_extreme_ratio = 0
            features.append(vol_extreme_ratio)
            
            # === 8. 新增趋势特征 (2个) ===
            # 高低点均值偏离
            if idx >= 10:
                hl_avg = (high + low) / 2
                hl_avg_10 = np.mean([(highs[idx-j] + lows[idx-j])/2 for j in range(10)])
                hl_deviation = safe_divide(hl_avg - hl_avg_10, hl_avg_10, 0)
            else:
                hl_deviation = 0
            features.append(hl_deviation)
            
            # 缺口强度（|Close - Close_prev| / Close_prev）
            if idx > 0:
                gap_strength = safe_divide(abs(close - prev_close), prev_close, 0)
            else:
                gap_strength = 0
            features.append(gap_strength)

            # === 9. 新增支撑阻力特征 (2个) ===

            # === F09_01: 距离MA5百分比 [P2-SUPPORT-02] ===
            # 定义：当前价格与MA5的距离百分比，正值表示在MA5上方（超买），负值表示在MA5下方（超卖）
            distance_ma5_percent = safe_divide(close - ma5, ma5, 0) * 100
            # 归一化到 [-1, 1] 的范围（假设±3%为合理范围）
            distance_ma5_percent_normalized = max(-1.0, min(1.0, distance_ma5_percent / 3.0))
            features.append(distance_ma5_percent_normalized)

            # === F09_02: MA5支撑强度 [P2-SUPPORT-03] ===
            # 定义：判断MA5是否被击穿及缓冲程度（以ATR计）
            atr_val = atr[idx]
            if low > ma5:
                # MA5未被击穿，有支撑
                ma5_support_strength = safe_divide(low - ma5, atr_val, 0)
                ma5_support_strength = min(ma5_support_strength, 2.0)  # 上限2倍ATR
            elif close > ma5:
                # MA5被击穿但收回来了
                ma5_support_strength = safe_divide(close - low, atr_val, 0)
                ma5_support_strength = min(ma5_support_strength, 1.0)  # 上限1倍ATR
            else:
                # MA5被彻底击穿
                ma5_support_strength = -1.0
            features.append(ma5_support_strength)

            # 验证特征数量
            if len(features) != NUM_FEATURES:
                print(f"⚠️  警告: 第{i+1}根K线特征数量不匹配! 期望{NUM_FEATURES}个，实际{len(features)}个")
            
            features_sequence.append(features)
        
        return np.array(features_sequence, dtype=np.float32)
    
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
        np.array: 特征序列 (60, 42)，数据不足120根时返回None
    """
    data = analyzer.get_training_data_smart(image_path, min_klines=120)
    
    if data is None or not data.get('kline_data'):
        return None
    
    return extract_features_sequence_from_kline_data(data['kline_data'])


# ============================================================================
# 特征分组信息（用于可解释性分析）
# ============================================================================
def get_feature_groups():
    """获取特征分组信息（56个特征）"""
    return {
        'price': {
            'name': '【第1组】价格均线特征（14个）',
            'features': [
                'F01_01: MA5价格归一化', 'F01_02: MA25价格归一化', 'F01_03: 价格>MA5', 'F01_04: 价格>MA25',
                'F01_05: MA5>MA25', 'F01_06: MA25趋势斜率', 'F01_07: K线收阳', 'F01_08: 价格相对MA25强度',
                'F01_09: K线方向强度', 'F01_10: 红绿K线比例',
                'F01_11: MA5-MA25粘合度 [P0]', 'F01_12: MA5-MA25发散速度 [P0]',
                'F01_13: K线实体率 [P1]', 'F01_14: K线影线强度 [P1]'
            ],
            'indices': list(range(0, 14))
        },
        'macd': {
            'name': '【第2组】MACD特征（10个）',
            'features': [
                'F02_01: DIF归一化', 'F02_02: DEA归一化', 'F02_03: MACD柱归一化', 'F02_04: DIF-DEA距离归一化',
                'F02_05: MACD金叉', 'F02_06: MACD死叉', 'F02_07: DIF变化率', 'F02_08: DEA变化率',
                'F02_09: DIF-DEA发散速度 [P0]', 'F02_10: MACD柱加速度 [P1]'
            ],
            'indices': list(range(14, 24))
        },
        'volume': {
            'name': '【第3组】成交量特征（12个）',
            'features': [
                'F03_01: MA5量归一化', 'F03_02: MA60量归一化', 'F03_03: 量线MA5>MA60', 'F03_04: MA5上方占比',
                'F03_05: 平均缩量幅度', 'F03_06: 成交量相对强度', 'F03_07: MA60量线趋势斜率', 'F03_08: 成交量波动率',
                'F03_09: 量线MA5-MA60粘合度 [P0]', 'F03_10: 量线MA5-MA60发散速度 [P0]',
                'F03_11: 成交量异常倍数 [P1]', 'F03_12: 价量同步性 [P1]'
            ],
            'indices': list(range(24, 36))
        },
        'volatility': {
            'name': '【第4组】波动率特征（4个）',
            'features': [
                'F04_01: ATR归一化', 'F04_02: 布林带位置', 'F04_03: 布林带宽度归一化', 'F04_04: 波动率加速度'
            ],
            'indices': list(range(36, 40))
        },
        'strength': {
            'name': '【第5组】相对强度特征（5个）',
            'features': [
                'F05_01: RSI归一化', 'F05_02: RSI快线 [P2]', 'F05_03: 价格动量20日',
                'F05_04: 价格位置60日', 'F05_05: 距离MA25百分比 [P2]'
            ],
            'indices': list(range(40, 45))
        },
        'momentum': {
            'name': '【第6组】强势特征（4个）',
            'features': [
                'F06_01: KDJ快线值', 'F06_02: KDJ_D线值 [P2]', 'F06_03: 价格加速度', 'F06_04: 连续K线强度'
            ],
            'indices': list(range(45, 49))
        },
        'volume_advanced': {
            'name': '【第7组】量能特征（3个）',
            'features': [
                'F07_01: 成交量加速度', 'F07_02: 价量相关性', 'F07_03: 成交量极值占比'
            ],
            'indices': list(range(49, 52))
        },
        'trend': {
            'name': '【第8组】趋势特征（2个）',
            'features': [
                'F08_01: 高低点均值偏离', 'F08_02: 缺口强度'
            ],
            'indices': list(range(52, 54))
        },
        'support': {
            'name': '【第9组】支撑阻力特征（2个）',
            'features': [
                'F09_01: 距离MA5百分比 [P2]', 'F09_02: MA5支撑强度 [P2]'
            ],
            'indices': list(range(54, 56))
        }
    }


# ============================================================================
# 特征统计信息
# ============================================================================
def print_feature_info():
    """打印特征信息"""
    print("=" * 90)
    print("AI训练特征系统信息（v3.3 - 56特征超强版）")
    print("=" * 90)
    print(f"\n📊 特征总数: {NUM_FEATURES}个")
    print("📚 特征来源: 原有46特征 + 新增10个（K线形态、MACD高级、成交量高级、支撑阻力）\n")

    groups = get_feature_groups()
    for group_key, group_info in groups.items():
        print(f"\n{group_info['name']}")
        for i, name in enumerate(group_info['features'], 1):
            idx = group_info['indices'][i-1]
            print(f"    {idx+1:2d}. {name}")

    print("\n" + "=" * 90)
    print("🎯 核心创新（纠缠状态系统 - P0优先级）:")
    print("  ★ F01_11/12: MA5-MA25粘合度 + 发散速度 → 反转预警")
    print("  ★ F02_09:    DIF-DEA发散速度 → MACD趋势强度")
    print("  ★ F03_09/10: 量线MA5-MA60粘合度 + 发散速度 → 资金活跃度变化")

    print("\n🔧 主要优化（P1-P2优先级）:")
    print("  ✓ F01_13/14: K线形态（实体率、影线强度） → 力度判断")
    print("  ✓ F02_10:    MACD柱加速度 → 动能加速")
    print("  ✓ F03_11/12: 成交量异常倍数、价量同步性 → 资金配合")
    print("  ✓ F05_02:    RSI快线(5) → 短期超买超卖")
    print("  ✓ F06_02:    KDJ_D线值 → 平稳随机指标")
    print("  ✓ F05_05/09_01/09_02: 支撑阻力距离百分比 → 突破难度")

    print("\n📝 技术说明:")
    print("  ✓ 所有特征使用连续归一化值（避免硬阈值）")
    print("  ✓ RSI采用标准EMA平滑（威尔德斯平均）")
    print("  ✓ KDJ采用标准随机指标（RSV平滑 + EMA K/D线 + J线）")
    print("  ✓ 特征编码：[组号]-[序号]，便于追踪和调试")
    print("  ✓ MA趋势斜率从10周期优化为25周期")
    print("  ✓ 消除代码冗余：统一均线窗口提取逻辑")
    print("  ✓ 统一边界条件：使用idx表示绝对索引，避免混淆")
    print("  ✓ 所有除法运算添加安全保护")
    print("=" * 80)


if __name__ == '__main__':
    print_feature_info()
