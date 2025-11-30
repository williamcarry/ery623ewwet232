# 未实现特征完整列表与实现指南

**文档目的：** 列出FEATURE_ANALYSIS.md和OPTIMIZATION_PLAN.md中所有未实现的特征  
**分类方式：** 按优先级(P2/P3) + 问题号 + 特征类别  
**总计：** 43个未实现特征（28个来自问题分析 + 15个来自优化方案）  
**文档日期：** 2025-01-15

---

## 📊 未实现特征总体统计

| 来源 | 优先级 | 特征数 | 特征码范围 | 预期收益 |
|------|--------|--------|----------|---------|
| FEATURE_ANALYSIS.md | P2+P3 | 28个 | 无（待分配） | +4-6% |
| OPTIMIZATION_PLAN.md | P2+P3 | 15个 | 无（待分配） | +2-3% |
| **合计** | - | **43个** | - | **+6-9%** |

---

## 🔴 第一部分：FEATURE_ANALYSIS.md未实现特征（28个）

### 问题1：均线体系不完善 ⭐⭐⭐ P2高优先级

**现状问题：**
- 当前仅有MA5、MA25
- 缺少MA10、MA60、MA120等常用均线
- 无法形成完整的多时间框架判断

**未实现特征（4个）：**

#### 1.1 MA10价格归一化
```
特征编码：（待分配，建议：F01_15）
特征名称：MA10价格归一化
计算方式：MA10 / Close
含义：短期支撑线相对强度
实现位置：extract_features_sequence_from_kline_data() 第370-396行附近

代码参考：
ma10_prices = np.convolve(closes, np.ones(10)/10, mode='valid')
feature_value = safe_divide(ma10, close, 1.0)
```

**业务用途：** 识别不同周期的趋势方向，补充MA5+MA25的缺口

---

#### 1.2 MA60价格归一化
```
特征编码：（待分配，建议：F01_16）
特征名称：MA60价格归一化
计算方式：MA60 / Close
含义：中长期趋势线相对强度
实现位置：extract_features_sequence_from_kline_data() 第370-396行附近

代码参考：
ma60_prices = np.convolve(closes, np.ones(60)/60, mode='valid')
feature_value = safe_divide(ma60, close, 1.0)
```

**业务用途：** 判断长期趋势，避免逆势交易

---

#### 1.3 均线粘合度（多均线排列）
```
特征编码：（待分配，建议：F01_17）
特征名称：均线粘合度（排列度指标）
计算方式：
if 5日 > 10日 > 25日 > 60日:
    粘合度 = 1.0  # 完美多头排列
elif |MA5-MA10| < threshold and |MA10-MA25| < threshold:
    粘合度 = (1 - |MA5-MA10|/MA25) * 0.5  # 粘合状态
else:
    粘合度 = 0  # 发散状态

含义：多个均线的排列状态，判断趋势强势度
实现位置：extract_features_sequence_from_kline_data() 第380-410行附近
```

**业务用途：** 识别强势上升、弱势横盘、转折点等信号

---

#### 1.4 MA60趋势斜率
```
特征编码：（待分配，建议：F01_18）
特征名称：MA60趋势斜率
计算方式：
if ma60_idx >= 25:
    ma60_slope = (ma60_prices[ma60_idx] - ma60_prices[ma60_idx-25]) / 25
    slope_normalized = safe_divide(ma60_slope, ma60, 0)
else:
    slope_normalized = 0

含义：长期均线的斜率，判断长期趋势方向和强度
实现位置：extract_features_sequence_from_kline_data() 第420-440行附近
```

**业务用途：** 识别长期趋势的加速或减速阶段

---

### 问题2：K线形态特征严重缺失 ⭐⭐⭐ P2高优先级

**现状问题：**
- 当前K线特征太简化（仅收阳、方向强度、比例）
- 无法识别反转形态

**未实现特征（4个）：**

#### 2.1 锤子线判断
```
特征编码：（待分配，建议：F01_19）
特征名称：锤子线判断
计算方式：
k_line_body = abs(close - open_price)
lower_shadow = min(open_price, close) - low
upper_shadow = high - max(open_price, close)

if lower_shadow > k_line_body * 2 and upper_shadow < k_line_body * 0.5:
    hammer_signal = 1  # 锤子线确认
else:
    hammer_signal = 0

含义：底部反转信号，下影线明显长于实体
实现位置：extract_features_sequence_from_kline_data() 第520-550行
```

**业务用途：** 识别底部反转机会，尤其在低位区域

---

#### 2.2 吞没形态判断
```
特征编码：（待分配，建议：F01_20）
特征名称：吞没形态判断
计算方式：
if idx > 0:
    prev_body = abs(closes[idx-1] - opens[idx-1])
    prev_high = highs[idx-1]
    prev_low = lows[idx-1]
    
    curr_body = abs(close - open_price)
    
    # 当前K线实体完全覆盖前一K线实体
    if (close > prev_high and open_price < prev_low) or \
       (open_price > prev_high and close < prev_low):
        engulfing_signal = 1
    else:
        engulfing_signal = 0
else:
    engulfing_signal = 0

含义：反转确认信号，当前K线完全覆盖前一K线
实现位置：extract_features_sequence_from_kline_data() 第520-560行
```

**业务用途：** 确认趋势反转，增加交易信号的可靠性

---

#### 2.3 十字星判断
```
特征编码：（待分配，建议：F01_21）
特征名称：十字星判断
计算方式：
k_line_body = abs(close - open_price)
true_range = high - low

if k_line_body < true_range * 0.1:  # 实体极小
    doji_signal = 1  # 十字星
else:
    doji_signal = 0

含义：多空均衡信号，小实体表示买卖双方势均力敌
实现位置：extract_features_sequence_from_kline_data() 第520-570行
```

**业务用途：** 识别变盘点，多空转换的临界位置

---

#### 2.4 上影线过长判断
```
特征编码：（待分配，建议：F01_22）
特征名称：上影线过长判断
计算方式：
atr_val = atr[idx]
upper_shadow = high - max(close, open_price)

if upper_shadow > atr_val * 1.5:
    long_upper_shadow = 1  # 上影线过长
else:
    long_upper_shadow = 0

含义：压力信号，空方压力强，可能回撤
实现位置：extract_features_sequence_from_kline_data() 第520-580行
```

**业务用途：** 识别压力位，预判回撤风险

---

### 问题3：成交量分析深度不足 ⭐⭐ P2中优先级

**现状问题：**
- 无法判断放量异常程度
- 缺少资金流向指标
- 缺少换手率等热度指标

**未实现特征（4个）：**

#### 3.1 OBV（能量潮）指标
```
特征编码：（待分配，建议：F03_13）
特征名称：OBV归一化
计算方式：
obv = np.zeros(len(closes))
obv[0] = volumes[0]

for i in range(1, len(closes)):
    if closes[i] > closes[i-1]:
        obv[i] = obv[i-1] + volumes[i]
    elif closes[i] < closes[i-1]:
        obv[i] = obv[i-1] - volumes[i]
    else:
        obv[i] = obv[i-1]

feature_value = (obv[idx] - obv[idx-20]) / (abs(obv[idx-20]) + 0.0001)

含义：累积能量方向，判断资金进出
实现位置：extract_features_sequence_from_kline_data() 后添加OBV计算
```

**业务用途：** 识别资金关键转折点，增强量价配合判断

---

#### 3.2 换手率相关特征
```
特征编码：（待分配，建议：F03_14）
特征名称：换手率相关特征
计算方式：
# 需要补充流通股数数据（这里假设为float(kline.circulating_shares)）
if circulating_shares > 0:
    turnover_rate = volume / circulating_shares * 100
    feature_value = min(turnover_rate / 10.0, 1.0)  # 归一化到[0,1]
else:
    feature_value = 0

含义：成交热度，换手率越高参与度越高
实现位置：需要在kline_data中补充流通股数信息
```

**业务用途：** 判断股票人气，识别爆发机会

---

#### 3.3 成交金额10日均线
```
特征编码：（待分配，建议：F03_15）
特征名称：成交金额10日均线
计算方式：
# 计算成交金额
trade_amount = close * volume

# 计算10日均线
ma10_amount = np.convolve(trade_amount, np.ones(10)/10, mode='valid')

# 当前金额相对均线
feature_value = safe_divide(close * volume, ma10_amount[idx], 1.0)

含义：单位资金流入量，判断资金规模变化
实现位置：extract_features_sequence_from_kline_data() 第426-487行附近
```

**业务用途：** 识别大资金进入/离场，预判趋势转折

---

#### 3.4 缩量反弹vs无量反弹识别
```
特征编码：（待分配，建议：F03_16）
特征名称：反弹质量判断
计算方式：
if idx >= 5:
    price_change = (close - closes[idx-5]) / closes[idx-5]  # 反弹幅度
    vol_change = (ma5_vol - ma5_volumes[ma5_idx-5]) / ma5_volumes[max(0, ma5_idx-5)]  # 成交量变化
    
    if price_change > 0:  # 反弹中
        if vol_change > 0.2:
            quality = 1  # 放量反弹（健康）
        elif vol_change > -0.2:
            quality = 0  # 温和反弹（一般）
        else:
            quality = -1  # 缩量反弹（弱势）
    else:
        quality = 0

含义：反弹质量评估，判断上升的可持续性
实现位置：extract_features_sequence_from_kline_data() 第490-520行
```

**业务用途：** 区分强势反弹和虚弱反弹

---

### 问题4：支撑阻力相关特征缺失 ⭐⭐ P2中优先级

**现状问题：**
- 仅有简单的相对位置特征
- 无法判断突破难度
- 缺少历史关键点判断

**未实现特征（3个）：**

#### 4.1 距离近期高点比例
```
特征编码：（待分配，建议：F04_05）
特征名称：距离近期高点比例
计算方式：
if idx >= 10:
    high_10 = np.max(highs[idx-9:idx+1])
    distance_to_high = (high_10 - close) / close * 100
    
    feature_value = -min(distance_to_high / 5.0, 1.0)  # 负值表示在高点下方
else:
    feature_value = 0

含义：离高点有多远，突破难度指标
实现位置：extract_features_sequence_from_kline_data() 第750-770行
```

**业务用途：** 判断突破压力大小，识别技术阻力位

---

#### 4.2 距离近期低点比例
```
特征编码：（待分配，建议：F04_06）
特征名称：距离近期低点比例
计算方式：
if idx >= 10:
    low_10 = np.min(lows[idx-9:idx+1])
    distance_to_low = (close - low_10) / close * 100
    
    feature_value = min(distance_to_low / 5.0, 1.0)  # 正值表示在低点上方
else:
    feature_value = 0

含义：反弹安全性，离低点支撑多远
实现位置：extract_features_sequence_from_kline_data() 第750-770行
```

**业务用途：** 判断反弹安全边际，识别技术支撑位

---

#### 4.3 前期密集成交区距离
```
特征编码：（待分配，建议：F04_07）
特征名称：前期密集成交区距离
计算方式：
# 这个需要分析成交量分布，识别密集成交价格区间
# 简化实现：使用价格在60日范围内的分布
if idx >= 60:
    price_window = closes[idx-59:idx+1]
    # 计算价格分布的众数区间（最常出现的价格范围）
    hist, bin_edges = np.histogram(price_window, bins=10)
    mode_bin = np.argmax(hist)
    mode_price = (bin_edges[mode_bin] + bin_edges[mode_bin+1]) / 2
    
    distance = abs(close - mode_price) / mode_price * 100
    feature_value = -min(distance / 5.0, 1.0)  # 离众数近=支撑强
else:
    feature_value = 0

含义：与历史成交集中区的距离，支撑强度指标
实现位置：extract_features_sequence_from_kline_data() 第750-780行
```

**业务用途：** 识别历史支撑/阻力位，预判反弹幅度

---

### 问题5：RSI/KDJ指标体系不完整 ⭐⭐ P2中优先级

**现状问题：**
- RSI仅单周期（14）
- KDJ仅有K线值
- 缺少指标背离、极值突破等高级用法

**未实现特征（4个）：**

#### 5.1 RSI背离判断
```
特征编码：（待分配，建议：F05_06）
特征名称：RSI背离判断
计算方式：
if idx >= 5:
    # 价格创新高但RSI未创新高 = 顶部背离信号
    price_is_high = close > np.max(closes[idx-5:idx])
    rsi_is_not_high = rsi[idx] < np.max(rsi[idx-5:idx])
    
    if price_is_high and rsi_is_not_high:
        bearish_divergence = 1  # 顶部背离（卖出信号）
    else:
        bearish_divergence = 0
    
    # 价格创新低但RSI未创新低 = 底部背离信号
    price_is_low = close < np.min(closes[idx-5:idx])
    rsi_is_not_low = rsi[idx] > np.min(rsi[idx-5:idx])
    
    if price_is_low and rsi_is_not_low:
        bullish_divergence = 1  # 底部背离（买入信号）
    else:
        bullish_divergence = 0
    
    feature_value = bullish_divergence - bearish_divergence  # [-1, 0, 1]
else:
    feature_value = 0

含义：价格与RSI的不同步现象，反转信号
实现位置：extract_features_sequence_from_kline_data() 第820-850行
```

**业务用途：** 增强顶底识别，提高反转交易的准确性

---

#### 5.2 RSI多周期共振
```
特征编码：（待分配，建议：F05_07）
特征名称：RSI多周期共振信号
计算方式：
# RSI快线和慢线已有，这里计算共振
rsi5_val = rsi5[idx]
rsi14_val = rsi[idx]

if rsi5_val > 70 and rsi14_val > 60:
    overbought_signal = 1  # 超买信号
elif rsi5_val < 30 and rsi14_val < 40:
    oversold_signal = -1  # 超卖信号
else:
    signal = 0

feature_value = overbought_signal + oversold_signal

含义：多周期RSI同向，强化超买超卖信号
实现位置：extract_features_sequence_from_kline_data() 第820-860行
```

**业务用途：** 识别强势超买/超卖状态，增强交易信号可靠性

---

#### 5.3 KDJ超买超卖判断
```
特征编码：（待分配，建议：F06_03）
特征名称：KDJ超买超卖判断
计算方式：
kdj_j = 3 * kdj_k[idx] - 2 * kdj_d[idx]  # J线计算

if kdj_j > 100:
    kdj_overbought = 1  # 极度超买
elif kdj_j < 0:
    kdj_oversold = -1  # 极度超卖
else:
    signal = 0

feature_value = kdj_overbought + kdj_oversold

含义：KDJ J线极值，反转信号
实现位置：extract_features_sequence_from_kline_data() 第880-900行
```

**业务用途：** 识别极端市场状态，提高反转交易概率

---

#### 5.4 KDJ快慢线交叉
```
特征编码：（待分配，建议：F06_04）
特征名称：KDJ快慢线交叉信号
计算方式：
kdj_d = kdj_d[idx]

if idx > 0:
    # K线从下穿过D线 = 金叉
    k_cross_above = kdj_k[idx] > kdj_d[idx] and kdj_k[idx-1] <= kdj_d[idx-1]
    
    # K线从上穿过D线 = 死叉
    k_cross_below = kdj_k[idx] < kdj_d[idx] and kdj_k[idx-1] >= kdj_d[idx-1]
    
    if k_cross_above:
        feature_value = 1  # 金叉（买入）
    elif k_cross_below:
        feature_value = -1  # 死叉（卖出）
    else:
        feature_value = 0
else:
    feature_value = 0

含义：KDJ交叉信号，短期趋势转折
实现位置：extract_features_sequence_from_kline_data() 第880-910行
```

**业务用途：** 识别短期趋势转折，配合K线形态增强信号

---

### 问题6：缺少价格结构特征 ⭐ P3低优先级

**现状问题：**
- 缺少底部形成强度
- 缺少趋势陡峭度
- 缺少加速度二阶导数

**未实现特征（3个）：**

#### 6.1 底部形成强度
```
特征编码：（待分配，建议：F08_03）
特征名称：底部形成强度
计算方式：
if idx >= 60:
    price_window = closes[idx-59:idx+1]
    low_60 = np.min(price_window)
    high_60 = np.max(price_window)
    
    # 底部区域定义：最低点附近20%价格范围内
    bottom_threshold = low_60 * 1.2
    bottom_count = np.sum(price_window < bottom_threshold)
    
    # 底部形成强度：底部区域出现频率
    bottom_strength = bottom_count / len(price_window)
else:
    bottom_strength = 0

feature_value = bottom_strength

含义：底部区域的坚实度，多次触及底部表示支撑强
实现位置：extract_features_sequence_from_kline_data() 第930-950行
```

**业务用途：** 识别坚实的底部形态，预测反弹幅度

---

#### 6.2 趋势陡峭度
```
特征编码：（待分配，建议：F08_04）
特征名称：趋势陡峭度
计算方式：
if idx >= 10:
    # 计算当前K线与MA25的夹角（相对速度）
    price_change = close - closes[idx-10]
    ma25_change = ma25 - ma25_prices[ma25_idx-10] if ma25_idx >= 10 else 0
    
    # 夹角：价格变化速度 / 均线变化速度
    trend_angle = safe_divide(price_change, ma25_change + 0.0001, 0)
    
    # 归一化：假设合理范围为[-2, 2]
    feature_value = max(-1.0, min(1.0, trend_angle / 2.0))
else:
    feature_value = 0

含义：价格上升与均线的夹角，判断趋势强度和可持续性
实现位置：extract_features_sequence_from_kline_data() 第930-960行
```

**业务用途：** 识别趋势加速、减速、反转的阶段

---

#### 6.3 价格加速度二阶导数
```
特征编码：（待分配，建议：F08_05）
特征名称：价格加速度二阶（加速度的加速度）
计算方式：
if idx >= 20:
    # 一阶导数：20日动量
    momentum_20 = (close - closes[idx-20]) / closes[idx-20]
    
    # 二阶导数：10日动量
    momentum_10 = (close - closes[idx-10]) / closes[idx-10]
    
    # 三阶导数：价格加速度的加速度
    if idx >= 30:
        momentum_30 = (close - closes[idx-30]) / closes[idx-30]
        acceleration_of_acceleration = (momentum_10 - momentum_20) - (momentum_30 - momentum_10)
    else:
        acceleration_of_acceleration = momentum_10 - momentum_20
    
    feature_value = max(-1.0, min(1.0, acceleration_of_acceleration / 0.1))
else:
    feature_value = 0

含义：动力的动力，判断加速度是否在加速/减速
实现位置：extract_features_sequence_from_kline_data() 第930-970行
```

**业务用途：** 识别趋势的质量变化，提前预警减速信号

---

### 问题7：代码层面优化 ⭐ P3低优先级（代码质量，非特征）

#### 7.1 均线计算丢失早期数据修复
**问题描述：**
```
当前代码：
ma5_prices = np.convolve(closes, np.ones(5)/5, mode='valid')
# mode='valid' 导致输出长度为 120 - 5 + 1 = 116
# 索引映射变得复杂，易出错
```

**改进方案：**
```python
# 改为 mode='same'，保持长度一致
ma5_prices = np.convolve(closes, np.ones(5)/5, mode='same')

# 或使用EMA替代，对齐数据
def calculate_ema_aligned(data, period):
    """EMA计算，长度对齐"""
    ema = np.zeros(len(data))
    alpha = 2.0 / (period + 1)
    ema[period-1] = np.mean(data[:period])
    for i in range(period, len(data)):
        ema[i] = ema[i-1] * (1 - alpha) + data[i] * alpha
    return ema
```

**影响范围：** 主要改进代码可读性和正确性，不影响特征值

---

#### 7.2 特征范围标准化统一
**问题描述：**
```python
# 当前特征范围不一致
'布林带位置' → [0, 1]
'价格相对MA25强度' → [-∞, +∞]  # 无界
'RSI归一化' → [0, 1]
'价格加速度' → 无界  # 可能很大
```

**改进方案：**
```python
def normalize_features_globally(features):
    """将所有特征归一化到 [-1, 1] 范围"""
    # 使用tanh或sigmoid确保有界性
    
    normalized = np.zeros_like(features)
    for i in range(features.shape[1]):  # 遍历每个特征
        col = features[:, i]
        
        # 计算该特征的标准差
        std = np.std(col)
        if std > 0:
            # 使用tanh映射到[-1, 1]
            normalized[:, i] = np.tanh(col / (std * 3))
        else:
            normalized[:, i] = 0
    
    return normalized

# 在训练前调用
features_train = normalize_features_globally(features_train)
features_test = normalize_features_globally(features_test)
```

**影响范围：** 提高模型训练稳定性，可能改进准确率0.5-1%

---

## 🟠 第二部分：OPTIMIZATION_PLAN.md未实现特征（15个）

### 优化1：深化MA5与MA25的交互关系（扩展）

#### 1.3 双均线动量（新增补充）
```
特征编码：（待分配，建议：F01_19）
特征名称：双均线动量
计算方式：
if ma5_idx > 0 and ma25_idx > 0:
    双均线动量 = (MA5 - MA25) / Close
    # 含义：
    # > 0：短期强于中期
    # < 0：中期强于短期
    # 变化率：动量方向

feature_value = safe_divide(ma5 - ma25, close, 0)

含义：MA5和MA25之间的相对强度
实现位置：extract_features_sequence_from_kline_data() 第600行附近
```

---

### 优化2：K线形态在MA背景下的识别（扩展）

#### 2.2 K线与MA25夹角（趋势陡峭度）
```
特征编码：（待分配，建议：F01_23）
特征名称：K线与MA25夹角（已在问题6.2实现过）

注意：这个特征可与问题6.2合并实现
```

#### 2.3 高级K线形态（追加4个）

##### 2.3.1 晨星形态
```
特征编码：（待分配，建议：F01_24）
特征名称：晨星形态判断
计算方式：
if idx >= 2:
    # 3根K线：第一根阴线，第二根小实体，第三根阳线
    k1_body = abs(closes[idx-2] - opens[idx-2])
    k2_body = abs(closes[idx-1] - opens[idx-1])
    k3_body = abs(close - open_price)
    
    # 晨星条件：
    # 1. 第一根K线是阴线，在MA下方
    # 2. 第二根K线实体小（十字星或小阴阳线）
    # 3. 第三根K线是阳线，收盘价在第一根K线上半部分
    
    condition1 = closes[idx-2] < opens[idx-2] and closes[idx-2] < ma25
    condition2 = k2_body < k1_body * 0.5
    condition3 = close > opens[idx-2] and close > opens[idx-2] + (opens[idx-2] - closes[idx-2]) / 2
    
    if condition1 and condition2 and condition3:
        morning_star = 1
    else:
        morning_star = 0
else:
    morning_star = 0

含义：底部反转形态，三根K线组成
实现位置：extract_features_sequence_from_kline_data() 第620-650行
```

---

##### 2.3.2 孤岛反转
```
特征编码：（待分配，建议：F01_25）
特征名称：孤岛反转判断
计算方式：
if idx >= 1:
    # 孤岛反转：缺口+反转+缺口
    # 1. 向下缺口（gap down）然后反转向上
    # 2. 或向上缺口（gap up）然后反转向下
    
    gap_down = low > closes[idx-1]  # 当前K线最低价在前根K线最高价之上
    gap_up = high < closes[idx-1]   # 当前K线最高价在前根K线最低价之下（错误）
    
    # 正确的缺口判断
    gap_down = opens[idx-1] > closes[idx-2]  # 向下缺口
    reversal_up = close > opens[idx-1]        # 反转向上
    
    if gap_down and reversal_up:
        island_reversal = 1
    else:
        island_reversal = 0
else:
    island_reversal = 0

含义：强反转信号，结合缺口和K线形态
实现位置：extract_features_sequence_from_kline_data() 第620-660行
```

---

### 优化3：成交量与MA5/MA60配合度（扩展）

#### 3.2 成交量斜率与价格方向
```
特征编码：（待分配，建议：F03_17）
特征名称：成交量斜率与价格方向配合
计算方式：
if idx >= 5 and ma60_vol_idx >= 5:
    # 成交量60日均线的斜率
    vol_slope = (ma60_volumes[ma60_vol_idx] - ma60_volumes[ma60_vol_idx-5]) / 5
    
    # 价格方向（5根K线变化）
    price_direction = 1 if close > closes[idx-5] else -1
    
    # 配合度：成交量是否顺应价格方向
    if vol_slope > 0 and price_direction > 0:
        sync_strength = 1  # 价涨量增（最强）
    elif vol_slope < 0 and price_direction < 0:
        sync_strength = 1  # 价跌量缩（正常）
    else:
        sync_strength = -1  # 背离（异常）
    
    feature_value = sync_strength
else:
    feature_value = 0

含义：成交量与价格的同步性判断
实现位置：extract_features_sequence_from_kline_data() 第760-780行
```

---

#### 3.3 连续放量根数
```
特征编码：（待分配，建议：F03_18）
特征名称：连续放量根数
计算方式：
if idx >= 5 and ma5_idx >= 5:
    # 统计最近10根中有多少根超过MA5_vol * 1.5
    consecutive_increasing_count = 0
    
    for j in range(5):
        vol_idx = ma5_idx - 5 + j
        if vol_idx >= 0 and ma5_volumes[vol_idx] > ma60_volumes[vol_idx] * 1.5:
            consecutive_increasing_count += 1
    
    feature_value = consecutive_increasing_count / 5.0  # 归一化到[0, 1]
else:
    feature_value = 0

含义：持续放量的持续性，连续放量根数多表示资金持续关注
实现位置：extract_features_sequence_from_kline_data() 第760-800行
```

---

### 优化4：支撑阻力进一步补充

#### 4.2 均线支撑强度（进阶版）
```
特征编码：（待分配，建议：F09_03）
特征名称：均线支撑强度（多重判断）
计算方式：
# MA5支撑强度已有(F09_02)，这里定义MA25的

if low > ma25:
    # MA25未被击穿，有支撑
    ma25_support = (low - ma25) / atr_val
    ma25_support = min(ma25_support, 2.0)
elif close > ma25:
    # MA25被击穿但收回来了
    ma25_support = (close - low) / atr_val * 0.5
    ma25_support = min(ma25_support, 1.0)
else:
    # MA25被彻底击穿
    ma25_support = -1.0

feature_value = ma25_support

含义：中期均线的支撑程度
实现位置：extract_features_sequence_from_kline_data() 第960-990行
```

---

### 优化5：RSI/KDJ多周期共振进一步扩展

这部分特征已在问题5中详细列出，避免重复

---

### 优化6：长期趋势背景（P3可选）

#### 6.1 长期趋势位置
```
特征编码：（待分配，建议：F08_06）
特征名称：长期趋势相对位置
计算方式：
if idx >= 120:
    # 120天前的价格（作为基准）
    price_120 = closes[idx - 120]
    
    # 120天的最高和最低
    price_window_120 = closes[idx-119:idx+1]
    high_120 = np.max(price_window_120)
    low_120 = np.min(price_window_120)
    
    # 当前价格在120天区间中的相对位置
    if high_120 > low_120:
        position_120 = (close - low_120) / (high_120 - low_120)
    else:
        position_120 = 0.5
    
    # 相对120天前的涨跌幅
    long_term_return = (close - price_120) / price_120
else:
    position_120 = 0.5
    long_term_return = 0

feature_value = position_120

含义：长期趋势中的相对位置，判断大方向
实现位置：extract_features_sequence_from_kline_data() 第990-1010行
```

---

## 📋 实现优先级建议

### 🔴 优先实现（P2高优先）
1. **问题1的4个特征**：均线体系（+2-3%）
2. **问题2的4个特征**：K线形态（+2-3%）
3. **问题3的前2个特征**：OBV + 换手率（+1-2%）
4. **优化1.3**：双均线动量（+0.5-1%）

**预期投入：** 2-3天  
**预期收益：** +5-9%准确率

### 🟠 次要实现（P2中优先）
5. **问题4的3个特征**：支撑阻力距离（+1-2%）
6. **问题5的4个特征**：RSI/KDJ高级用法（+1-2%）
7. **优化2.3的2个特征**：晨星 + 孤岛反转（+1-2%）

**预期投入：** 2-3天  
**预期收益：** +3-6%准确率

### 🟡 可选实现（P3低优先）
8. **问题6的3个特征**：价格结构（+0.5-1%）
9. **问题7**：代码优化（代码质量提升）
10. **优化6**：长期趋势（+0.5-1%）

**预期投入：** 1-2天  
**预期收益：** +1-3%准确率

---

## 🛠️ 实现检查清单

### 添加特征前的准备
- [ ] 为新特征分配编码（F##_##格式）
- [ ] 在FEATURE_NAMES数组中添加特征名称
- [ ] 在get_feature_groups()中添加特征分组信息
- [ ] 为特征添加详细注释说明计算逻辑

### 实现时的注意事项
- [ ] 使用safe_divide()避免除以零
- [ ] 所有特征都应归一化到[-1, 1]或[0, 1]范围
- [ ] 添加边界条件检查（idx >= period）
- [ ] 测试特征值的范围是否符合预期
- [ ] 更新NUM_FEATURES常量

### 实现后的验证
- [ ] 运行print_feature_info()验证特征列表
- [ ] 在测试数据上提取特征，检查有无NaN或inf
- [ ] 比较新旧特征提取的性能差异
- [ ] 在模型训练上验证准确率提升

---

## 📈 完整实现预期

| 阶段 | 特征数 | 代码改动行数 | 预期时间 | 预期收益 |
|------|--------|----------|---------|---------|
| **当前（已完成）** | 56个 | ~1000 | - | 基础 |
| **阶段1（P2高优）** | 12个 | ~800 | 2-3天 | +5-9% |
| **阶段2（P2中）** | 12个 | ~1000 | 2-3天 | +3-6% |
| **阶段3（P3低）** | 4个 | ~500 | 1-2天 | +1-3% |
| **合计** | **84个** | **~3300** | **5-8天** | **+9-18%** |

---

**文档完成日期：** 2025-01-15  
**维护者：** AI特征提取系统  
**最后更新：** 2025-01-15
