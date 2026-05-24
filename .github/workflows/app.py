import streamlit as st
import numpy as np
from scipy.optimize import linprog

# 设置页面标题
st.set_page_config(page_title="博弈论全功能计算器", layout="centered")

# --- 在最前端加入署名 ---
st.title(" 矩阵对策综合求解计算器")
st.caption("✨ **专硕39-彭浩翔 制**") 
st.write("---")

st.markdown("系统将严格按照以下顺序，分步求解：\n**1. 优势简化** ➔ **2. 纯策略鞍点检验** ➔ **3. 混合策略求解**")

# 1. 动态设置矩阵大小
col1, col2 = st.columns(2)
with col1:
    rows = st.number_input("原始矩阵行数 (A 的策略数)", min_value=2, max_value=8, value=3, step=1)
with col2:
    cols = st.number_input("原始矩阵列数 (B 的策略数)", min_value=2, max_value=8, value=3, step=1)

st.write("---")
st.subheader(" 填写原始矩阵数值")
st.caption("注：数值代表行玩家 A 的收益（列玩家 B 的损失）。")

# 2. 构建动态输入表格
matrix_data = []
for i in range(rows):
    row_cells = st.columns(cols)
    row_values = []
    for j in range(cols):
        with row_cells[j]:
            # 默认填充一个适合演示优势简化的矩阵
            if i == 2: default_val = 0.0
            elif j == 2: default_val = 10.0
            else: default_val = float(i * cols + j + 2)
            
            val = st.number_input(f"A{i+1}, B{j+1}", value=default_val, key=f"cell_{i}_{j}")
            row_values.append(val)
    matrix_data.append(row_values)

matrix = np.array(matrix_data)

st.write("---")
st.subheader(" 原始博弈矩阵")
st.dataframe(matrix)

# 3. 核心算法：步骤 1 - 优势简化法 (Dominance Elimination)
remaining_rows = list(range(rows))
remaining_cols = list(range(cols))
history_log = [] 
eliminated = True

while eliminated:
    eliminated = False
    current_matrix = matrix[np.ix_(remaining_rows, remaining_cols)]
    curr_r, curr_c = current_matrix.shape
    
    # 检查行占优
    if curr_r > 1:
        row_to_remove = None
        for i in range(curr_r):
            for j in range(curr_r):
                if i != j:
                    if np.all(current_matrix[j, :] >= current_matrix[i, :]) and np.any(current_matrix[j, :] > current_matrix[i, :]):
                        row_to_remove = i
                        break
            if row_to_remove is not None:
                break
        
        if row_to_remove is not None:
            actual_row_idx = remaining_rows[row_to_remove]
            history_log.append(f" 选手 A 删除了劣势策略 **A{actual_row_idx+1}**（其收益被其他策略完全压制）")
            remaining_rows.pop(row_to_remove)
            eliminated = True
            continue 

    # 检查列占优
    if curr_c > 1:
        col_to_remove = None
        for i in range(curr_c):
            for j in range(curr_c):
                if i != j:
                    if np.all(current_matrix[:, j] <= current_matrix[:, i]) and np.any(current_matrix[:, j] < current_matrix[:, i]):
                        col_to_remove = i
                        break
            if col_to_remove is not None:
                break
        
        if col_to_remove is not None:
            actual_col_idx = remaining_cols[col_to_remove]
            history_log.append(f" 选手 B 删除了劣势策略 **B{actual_col_idx+1}**（其造成的损失比其他策略都大）")
            remaining_cols.pop(col_to_remove)
            eliminated = True
            continue

reduced_matrix = matrix[np.ix_(remaining_rows, remaining_cols)]

st.write("---")
st.subheader(" 步骤 1：优势简化法处理结果")
if history_log:
    for log in history_log:
        st.markdown(log)
    st.markdown("** 化简后的最终矩阵：**")
    st.dataframe(reduced_matrix)
else:
    st.info("经检查，当前矩阵不存在可以被严格简化的优势/劣势策略。保持原矩阵进行后续计算。")

# 4. 核心算法：步骤 2 - 纯策略鞍点检验
st.write("---")
st.subheader(" 步骤 2：纯策略鞍点（Saddle Point）检验")

r_mins = np.min(reduced_matrix, axis=1)
c_maxs = np.max(reduced_matrix, axis=0)
max_min = np.max(r_mins)
min_max = np.min(c_maxs)

saddle_points = []
red_rows, red_cols = reduced_matrix.shape
for i in range(red_rows):
    for j in range(red_cols):
        if reduced_matrix[i, j] == r_mins[i] and reduced_matrix[i, j] == c_maxs[j]:
            saddle_points.append((i, j, reduced_matrix[i, j]))

st.write(f"**行最小值中的最大值 (Max-Min / 选手 A 的保守收益):** `{max_min}`")
st.write(f"**列最大值中的最小值 (Min-Max / 选手 B 的保守损失):** `{min_max}`")

has_saddle = len(saddle_points) > 0

if has_saddle:
    st.success(f" 检验结论：找到纯策略鞍点！该博弈存在纯策略纳什均衡。")
    for idx, (r, c, val) in enumerate(saddle_points):
        orig_r = remaining_rows[r] + 1
        orig_c = remaining_cols[c] + 1
        st.info(f"** 鞍点 {idx+1}:** 位置对应原始矩阵的第 **{orig_r}** 行，第 **{orig_c}** 列，博弈值为 **{val}**")
    st.markdown("> **说明：** 因为存在纯策略鞍点，双方只需各自坚定选择该固定策略即可。")
else:
    st.warning(f" 检验结论：该矩阵中**没有纯策略鞍点**。")
    st.markdown(f"> **判定依据：** 因为 $Max-Min ({max_min}) \\neq Min-Max ({min_max})$，说明双方无法在纯策略下达成妥协。")

# 5. 核心算法：步骤 3 - 混合策略求解
st.write("---")
st.subheader(" 步骤 3：混合策略求解（基于线性规划）")

if has_saddle:
    st.info(" 由于步骤 2 已找到纯策略鞍点，混合策略等同于以 100% 的概率选择该鞍点。")
    col_p, col_q = st.columns(2)
    with col_p:
        st.markdown("**选手 A 策略概率：**")
        for i in range(rows):
            is_saddle_row = any(remaining_rows[sp[0]] == i for sp in saddle_points)
            prob = 1.0 if is_saddle_row else 0.0
            st.write(f"策略 **A{i+1}** : `{prob*100:.2f}%`")
            st.progress(prob)
    with col_q:
        st.markdown("**选手 B 策略概率：**")
        for j in range(cols):
            is_saddle_col = any(remaining_cols[sp[1]] == j for sp in saddle_points)
            prob = 1.0 if is_saddle_col else 0.0
            st.write(f"策略 **B{j+1}** : `{prob*100:.2f}%`")
            st.progress(prob)
            
    # 纯策略情况下的最终行融合输出
    st.write("---")
    st.success(f" 决策总结：此博弈为纯策略对策，最终的期望收益受益值固定为 **{saddle_points[0][2]:.3f}**。")
else:
    st.markdown("由于不存在纯策略鞍点，系统已通过线性规划计算双方的最优随机概率分布：")
    
    # 线性规划求解（平移机制防止负数）
    shift = abs(np.min(reduced_matrix)) + 1 if np.min(reduced_matrix) <= 0 else 0
    M_pos = reduced_matrix + shift
    
    # 选手 A 线性规划
    res_a = linprog(np.ones(red_rows), A_ub=-M_pos.T, b_ub=-np.ones(red_cols), bounds=(0, None), method='highs')
    # 选手 B 线性规划
    res_b = linprog(-np.ones(red_cols), A_ub=M_pos, b_ub=np.ones(red_rows), bounds=(0, None), method='highs')

    if res_a.success and res_b.success:
        V = (1.0 / res_a.fun) - shift
        p_sub = res_a.x / np.sum(res_a.x)
        q_sub = res_b.x / np.sum(res_b.x)
        
        col_p, col_q = st.columns(2)
        with col_p:
            st.markdown("**选手 A 最优概率分配（映射回原始策略）：**")
            for i in range(rows):
                if i in remaining_rows:
                    idx = remaining_rows.index(i)
                    st.write(f"策略 **A{i+1}** : `{p_sub[idx]*100:.2f}%`")
                    st.progress(float(p_sub[idx]))
                else:
                    st.write(f"策略 **A{i+1}** : `0.00%` (已被步骤1优势法淘汰)")
                    st.progress(0.0)
                    
        with col_q:
            st.markdown("**选手 B 最优概率分配（映射回原始策略）：**")
            for j in range(cols):
                if j in remaining_cols:
                    idx = remaining_cols.index(j)
                    st.write(f"策略 **B{j+1}** : `{q_sub[idx]*100:.2f}%`")
                    st.progress(float(q_sub[idx]))
                else:
                    st.write(f"策略 **B{j+1}** : `0.00%` (已被步骤1优势法淘汰)")
                    st.progress(0.0)
                    
        # 混合策略情况下的最后一行融合输出
        st.write("---")
        st.success(f" 决策总结：在双方执行上述最优混合对策概率时，系统算得该博弈的**最终期望受益值（博弈值）为 {V:.3f}**。选手 A 按照此比例随机出牌，可确保长期平均收益稳定在该值，不受选手 B 策略变动的影响。")
    else:
        st.error(" 混合策略线性规划求解失败，请检查矩阵数值是否合理。")
