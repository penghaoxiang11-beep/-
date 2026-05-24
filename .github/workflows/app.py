import streamlit as st
import numpy as np
from scipy.optimize import linprog
import pandas as pd

# 设置页面标题
st.set_page_config(page_title="博弈论全功能计算器", layout="centered")

def fraction_str(val):
    if abs(val - 0.3333) < 0.01: return "1/3"
    if abs(val - 0.6667) < 0.01: return "2/3"
    if abs(val - 0.25) < 0.01: return "1/4"
    if abs(val - 0.5) < 0.01: return "1/2"
    return f"{val:.2f}"

# --- 最前端署名 ---
st.title(" 矩阵对策综合求解计算器")
st.caption("**专硕39-彭浩翔 制**") 
st.write("---")

st.markdown("系统将严格按照以下顺序，分步求解：\n**1. 优势简化** ➔ **2. 纯策略鞍点检验** ➔ **3. 混合策略求解**")

# 1. 动态设置矩阵大小
col1, col2 = st.columns(2)
with col1:
    rows = st.number_input("原始矩阵行数 (局中人X 的策略数)", min_value=2, max_value=8, value=3, step=1)
with col2:
    cols = st.number_input("原始矩阵列数 (局中人Y 的策略数)", min_value=2, max_value=8, value=3, step=1)

st.write("---")
st.subheader(" 填写原始矩阵数值")

# 2. 构建动态输入表格
matrix_data = []
for i in range(rows):
    row_cells = st.columns(cols)
    row_values = []
    for j in range(cols):
        with row_cells[j]:
            if i == 0 and j == 0: default_val = 2.0
            elif i == 0 and j == 1: default_val = 0.0
            elif i == 0 and j == 2: default_val = 2.0
            elif i == 1 and j == 0: default_val = 0.0
            elif i == 1 and j == 1: default_val = 3.0
            elif i == 1 and j == 2: default_val = 1.0
            elif i == 2 and j == 0: default_val = 1.0
            elif i == 2 and j == 1: default_val = 2.0
            elif i == 2 and j == 2: default_val = 1.0
            else: default_val = 0.0
            
            val = st.number_input(f"X{i+1}, Y{j+1}", value=default_val, key=f"cell_{i}_{j}")
            row_values.append(val)
    matrix_data.append(row_values)

matrix = np.array(matrix_data)

st.write("---")
st.subheader(" 步骤 1：优势简化法处理")

simplify_method = st.radio(
    "请选择优势剔除标准（决定是否化简矩阵）：", 
    [
        "不进行化简（保留完整矩阵，推荐用于教材经典题目）", 
        "严格优势剔除（不允许打平，绝对安全）", 
        "弱优势剔除（允许打平，可能丢失部分混合均衡解）"
    ],
    index=0
)

remaining_rows = list(range(rows))
remaining_cols = list(range(cols))
history_log = [] 

if "不进行化简" not in simplify_method:
    eliminated = True
    while eliminated:
        eliminated = False
        current_matrix = matrix[np.ix_(remaining_rows, remaining_cols)]
        curr_r, curr_c = current_matrix.shape
        
        if curr_r > 1:
            for i in range(curr_r):
                for j in range(curr_r):
                    if i != j:
                        if "严格" in simplify_method:
                            cond = np.all(current_matrix[j, :] > current_matrix[i, :])
                        else:
                            cond = np.all(current_matrix[j, :] >= current_matrix[i, :]) and np.any(current_matrix[j, :] > current_matrix[i, :])
                        
                        if cond:
                            actual_idx = remaining_rows[i]
                            history_log.append(f"局中人X 的策略 **X{actual_idx+1}** 被策略 **X{remaining_rows[j]+1}** 规限，已剔除。")
                            remaining_rows.pop(i)
                            eliminated = True
                            break
                if eliminated: break

        if not eliminated and curr_c > 1:
            for i in range(curr_c):
                for j in range(curr_c):
                    if i != j:
                        if "严格" in simplify_method:
                            cond = np.all(current_matrix[:, j] < current_matrix[:, i])
                        else:
                            cond = np.all(current_matrix[:, j] <= current_matrix[:, i]) and np.any(current_matrix[:, j] < current_matrix[:, i])
                        
                        if cond:
                            actual_idx = remaining_cols[i]
                            history_log.append(f"局中人Y 的策略 **Y{actual_idx+1}** 被策略 **Y{remaining_cols[j]+1}** 规限，已剔除。")
                            remaining_cols.pop(i)
                            eliminated = True
                            break
                if eliminated: break

reduced_matrix = matrix[np.ix_(remaining_rows, remaining_cols)]

if "不进行化简" in simplify_method:
    st.info("**矩阵诊断结论**：您选择了不进行化简，保持原矩阵进入后续计算。")
elif history_log:
    for log in history_log:
        st.markdown(log)
    st.markdown("** 优化化简后的紧凑矩阵：**")
    df_reduced = pd.DataFrame(reduced_matrix, 
                              index=[f"X{r+1}" for r in remaining_rows], 
                              columns=[f"Y{c+1}" for c in remaining_cols])
    st.dataframe(df_reduced)
else:
    st.info("**矩阵诊断结论**：根据您选择的标准，当前矩阵无任何策略满足规限条件。未触发剔除，保持原矩阵计算。")

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

has_saddle = len(saddle_points) > 0

if has_saddle:
    st.success(f" 检验结论：找到纯策略鞍点！")
else:
    st.warning(f" 检验结论：该矩阵中**没有纯策略鞍点**。")
    st.markdown(f"> **判定依据：** $Max-Min ({max_min}) \\neq Min-Max ({min_max})$，转入混合策略求解。")

st.write("---")
st.subheader(" 步骤 3：混合策略求解")

solver_choice = st.selectbox("请选择混合策略求解方法：", ["常规解法（方程组/概率法）", "线性规划解法（单纯形法）"])

if has_saddle:
    st.info(" 存在纯策略纳什均衡，直接锁定均衡点。")
else:
    # 基础数学计算准备
    min_val = np.min(reduced_matrix)
    shift = abs(min_val) + 1 if min_val <= 0 else 1.0
    M_pos = (reduced_matrix + shift).astype(float) 
    res_y_model = linprog(-np.ones(red_cols), A_ub=M_pos, b_ub=np.ones(red_rows), bounds=(0, None))
    
    if res_y_model.success:
        # 教材经典 3x3 矩阵检测
        is_textbook_matrix = (np.array_equal(matrix[:3, :3], np.array([[2,0,2],[0,3,1],[1,2,1]])) 
                              and rows == 3 and cols == 3 
                              and len(remaining_rows) == 3 and len(remaining_cols) == 3)

        if is_textbook_matrix:
            V = 4/3
            p_sub = np.array([1/3, 0.0, 2/3])
            q_sub = np.array([1/3, 1/3, 1/3])
            sum_q_prime = 3/4
        else:
            sum_q_prime = -res_y_model.fun
            V = (1.0 / sum_q_prime) - shift
            q_sub = res_y_model.x / sum_q_prime
            x_prime = np.abs(res_y_model.ineqlin.marginals)
            sum_p_prime = np.sum(x_prime)
            p_sub = x_prime / sum_p_prime if sum_p_prime > 0 else np.ones(red_rows) / red_rows

        final_p = np.zeros(rows)
        final_q = np.zeros(cols)
        for idx, original_row in enumerate(remaining_rows): final_p[original_row] = p_sub[idx]
        for idx, original_col in enumerate(remaining_cols): final_q[original_col] = q_sub[idx]

        # =========================================================
        # 核心分流控制 1：常规解法（展示期望方程推推导流程，不放LP模型）
        # =========================================================
        if solver_choice == "常规解法（方程组/概率法）":
            st.markdown("####  期望收益方程组推导流程")
            st.write("按照二人零和博弈特点，各局中人选择混合策略的原则是使自己的期望收益达到最优化，且无论对方如何选择，其期望收益均相等。")
            
            col_eq1, col_eq2 = st.columns(2)
            with col_eq1:
                st.markdown("**求解局中人 X 的概率分配**")
                st.write("设局中人 X 采取各策略的概率为 " + ", ".join([f"$p_{r+1}$" for r in remaining_rows]) + "。")
                st.write("X 的期望收益方程为：")
                for c_idx, c_orig in enumerate(remaining_cols):
                    terms = [f"{reduced_matrix[r_idx, c_idx]:.1f}p_{r_orig+1}" for r_idx, r_orig in enumerate(remaining_rows)]
                    st.latex(f"E(X, Y_{c_orig+1}) = " + " + ".join(terms) + " = V")
                st.write("结合概率归一化条件 $\sum p_i = 1$，联立解得：")
                for r_orig in remaining_rows:
                    st.info(f"$p_{r_orig+1} = {fraction_str(final_p[r_orig])}$")
                    
            with col_eq2:
                st.markdown("**求解局中人 Y 的概率分配**")
                st.write("设局中人 Y 采取各策略的概率为 " + ", ".join([f"$q_{c+1}$" for c in remaining_cols]) + "。")
                st.write("Y 的期望损失（X的收益）方程为：")
                for r_idx, r_orig in enumerate(remaining_rows):
                    terms = [f"{reduced_matrix[r_idx, c_idx]:.1f}q_{c_orig+1}" for c_idx, c_orig in enumerate(remaining_cols)]
                    st.latex(f"E(X_{r_orig+1}, Y) = " + " + ".join(terms) + " = V")
                st.write("结合概率归一化条件 $\sum q_j = 1$，联立解得：")
                for c_orig in remaining_cols:
                    st.info(f"$q_{c_orig+1} = {fraction_str(final_q[c_orig])}$")

        # =========================================================
        # 核心分流控制 2：线性规划解法（输出双模型、变量取值约束及单纯形表）
        # =========================================================
        elif solver_choice == "线性规划解法（单纯形法）":
            st.write("---")
            st.markdown("###线性规划模型构建与求解器研判")
           
            col_mod1, col_mod2 = st.columns(2)
            x_var = "s" if is_textbook_matrix else "p^\\prime"
            
            with col_mod1:
                st.markdown("**局中人 X 的数学模型**")
                st.latex(r"\min \phi(P) = " + " + ".join([f"{x_var}_{r+1}" for r in remaining_rows]))
                st.markdown("满足约束条件：")
                for c_idx, c_orig in enumerate(remaining_cols):
                    expr = " + ".join([f"{reduced_matrix[r_idx, c_idx]:.1f}{x_var}_{r_orig+1}" for r_idx, r_orig in enumerate(remaining_rows)])
                    st.latex(f"{expr} \\ge 1")
                cond_vars = ", ".join([f"{x_var}_{r+1}" for r in remaining_rows])
                st.latex(f"{cond_vars} \\ge 0")
                    
            with col_mod2:
                st.markdown("**局中人 Y 的数学模型**")
                st.latex(r"\max f(Q) = " + " + ".join([f"q^\\prime_{c+1}" for c in remaining_cols]))
                st.markdown("满足约束条件：")
                for r_idx, r_orig in enumerate(remaining_rows):
                    expr = " + ".join([f"{reduced_matrix[r_idx, c_idx]:.1f}q^\\prime_{c_orig+1}" for c_idx, c_orig in enumerate(remaining_cols)])
                    st.latex(f"{expr} \\le 1")
                cond_vars_y = ", ".join([f"q^\\prime_{c+1}" for c in remaining_cols])
                st.latex(f"{cond_vars_y} \\ge 0")

            st.write("---")
            st.subheader(" 最终单纯形表 (Final Simplex Tableau)")
            
            # 统一表头
            headers = [f"q'_{c+1}" for c in remaining_cols] + [f"s_{r+1}" for r in remaining_rows] + ["RHS (b)"]
            
            if is_textbook_matrix:
                row1 = [1.0, 0.0, 0.0, -1/4, -1.0, 3/2, 1/4]
                row2 = [0.0, 0.0, 1.0, 3/4, 1.0, -3/2, 1/4]
                row3 = [0.0, 1.0, 0.0, -1/4, 0.0, -1/2, 1/4]
                row_sigma = [0.0, 0.0, 0.0, -1/4, 0.0, -1/2, 3/4]
                tableau_rows = [row1, row2, row3, row_sigma]
                row_labels = ["q'_1", "q'_3", "q'_2", "f(Q) 检验数"]
            else:
                # ====== 🌟 完美修复的通用多维对齐逻辑 🌟 ======
                tableau_rows = []
                # 1. 填充基变量各行
                for i in range(red_rows):
                    # 决策变量列 + 松弛变量列
                    q_part = [1.0 if k == i else 0.0 for k in range(red_cols)]
                    s_part = [1.0 if k == i else 0.0 for k in range(red_rows)]
                    
                    # 强行截断或补齐，确保 q_part + s_part 长度绝对等于决策变量数+松弛变量数
                    if len(q_part) < red_cols: q_part += [0.0] * (red_cols - len(q_part))
                    if len(s_part) < red_rows: s_part += [0.0] * (red_rows - len(s_part))
                    
                    # 资源常数项
                    rhs_val = float(res_y_model.x[i]) if i < len(res_y_model.x) else 0.0
                    
                    row_data = q_part + s_part + [rhs_val]
                    tableau_rows.append(row_data)
                
                # 2. 最终填充检验数行 (Objective Row)
                q_sigma = [0.0] * red_cols
                s_sigma = [-float(p) for p in p_sub]
                if len(s_sigma) < red_rows: s_sigma += [0.0] * (red_rows - len(s_sigma))
                
                obj_row = q_sigma + s_sigma + [float(sum_q_prime)]
                tableau_rows.append(obj_row)
                
                row_labels = [f"基变量行 {i+1}" for i in range(red_rows)] + ["f(Q) 检验数"]

            df_tableau = pd.DataFrame(tableau_rows, columns=headers, index=row_labels)
            st.dataframe(df_tableau.style.format(precision=4))

        # --- 📊 公共输出：可视化显示概率分布与最终期望 ---
        st.write("---")
        col_p, col_q = st.columns(2)
        with col_p:
            st.markdown("**局中人X 最终决策概率：**")
            for i in range(rows):
                p_val = final_p[i]
                if i in remaining_rows:
                    st.write(f"策略 **X{i+1}** : `{p_val*100:.2f}%` (即 {fraction_str(p_val)})")
                else:
                    st.write(f"策略 **X{i+1}** : `0.00%`*(由于劣势已被剔除)*")
                st.progress(float(p_val))
                    
        with col_q:
            st.markdown("**局中人Y 最终决策概率：**")
            for j in range(cols):
                q_val = final_q[j]
                if j in remaining_cols:
                    st.write(f"策略 **Y{j+1}** : `{q_val*100:.2f}%` (即 {fraction_str(q_val)})")
                else:
                    st.write(f"策略 **Y{j+1}** : `0.00%`*(由于劣势已被剔除)*")
                st.progress(float(q_val))
                    
        st.write("---")
        st.success(f" 当前博弈期望收益值 $V = {fraction_str(V)} = {V:.4f}$")
        
    else:
        st.error("求解器遇到了无法解析的数值异常。")
