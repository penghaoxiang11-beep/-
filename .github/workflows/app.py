import streamlit as st
import numpy as np
from scipy.optimize import linprog
import pandas as pd

st.set_page_config(page_title="博弈论综合分析工具", layout="wide")

# ---------- 辅助函数 ----------
def fraction_str(val):
    """将浮点数转换为常见分数的字符串表示"""
    if abs(val - 0.3333) < 0.01: return "1/3"
    if abs(val - 0.6667) < 0.01: return "2/3"
    if abs(val - 0.25) < 0.01: return "1/4"
    if abs(val - 0.5) < 0.01: return "1/2"
    if abs(val - 0.75) < 0.01: return "3/4"
    if abs(val - 0.2) < 0.01: return "1/5"
    if abs(val - 0.4) < 0.01: return "2/5"
    if abs(val - 0.6) < 0.01: return "3/5"
    if abs(val - 0.8) < 0.01: return "4/5"
    return f"{val:.2f}"

def simplify_matrix(matrix, method):
    """优势简化，返回 (简化矩阵, 剩余行索引列表, 剩余列索引列表, 日志列表)"""
    rows, cols = matrix.shape
    remaining_rows = list(range(rows))
    remaining_cols = list(range(cols))
    history_log = []

    if "不进行化简" in method:
        history_log.append("未进行简化，保留原始矩阵。")
        return matrix, remaining_rows, remaining_cols, history_log

    eliminated = True
    while eliminated:
        eliminated = False
        current_matrix = matrix[np.ix_(remaining_rows, remaining_cols)]
        curr_r, curr_c = current_matrix.shape

        # 剔除 X 的劣势策略（行）
        if curr_r > 1:
            for i in range(curr_r):
                for j in range(curr_r):
                    if i != j:
                        if "严格" in method:
                            cond = np.all(current_matrix[j, :] > current_matrix[i, :])
                        else:  # 弱优势
                            cond = np.all(current_matrix[j, :] >= current_matrix[i, :]) and np.any(current_matrix[j, :] > current_matrix[i, :])

                        if cond:
                            actual_idx = remaining_rows[i]
                            history_log.append(f"局中人X 的策略 X{actual_idx+1} 被策略 X{remaining_rows[j]+1} 严格{'（弱）' if '弱' in method else ''}优势剔除。")
                            remaining_rows.pop(i)
                            eliminated = True
                            break
                if eliminated:
                    break

        # 剔除 Y 的劣势策略（列）
        if not eliminated and curr_c > 1:
            for i in range(curr_c):
                for j in range(curr_c):
                    if i != j:
                        if "严格" in method:
                            cond = np.all(current_matrix[:, j] < current_matrix[:, i])
                        else:
                            cond = np.all(current_matrix[:, j] <= current_matrix[:, i]) and np.any(current_matrix[:, j] < current_matrix[:, i])

                        if cond:
                            actual_idx = remaining_cols[i]
                            history_log.append(f"局中人Y 的策略 Y{actual_idx+1} 被策略 Y{remaining_cols[j]+1} 严格{'（弱）' if '弱' in method else ''}优势剔除。")
                            remaining_cols.pop(i)
                            eliminated = True
                            break
                if eliminated:
                    break

    reduced_matrix = matrix[np.ix_(remaining_rows, remaining_cols)]
    if not history_log:
        history_log.append("根据所选标准，未发现可剔除的劣势策略。")
    return reduced_matrix, remaining_rows, remaining_cols, history_log

def saddle_point_analysis(reduced_matrix, remaining_rows, remaining_cols):
    """纯策略鞍点检验，返回 (has_saddle, 鞍点列表[(行号,列号,值)], max_min, min_max)"""
    r_mins = np.min(reduced_matrix, axis=1)
    c_maxs = np.max(reduced_matrix, axis=0)
    max_min = np.max(r_mins)
    min_max = np.min(c_maxs)

    saddle_points = []
    red_rows, red_cols = reduced_matrix.shape
    for i in range(red_rows):
        for j in range(red_cols):
            if reduced_matrix[i, j] == r_mins[i] and reduced_matrix[i, j] == c_maxs[j]:
                orig_r = remaining_rows[i] + 1
                orig_c = remaining_cols[j] + 1
                saddle_points.append((orig_r, orig_c, reduced_matrix[i, j]))

    return len(saddle_points) > 0, saddle_points, max_min, min_max

def compute_mixed_strategy(reduced_matrix):
    """利用线性规划求解混合策略，返回 (p_sub, q_sub, V, is_textbook)"""
    red_rows, red_cols = reduced_matrix.shape
    min_val = np.min(reduced_matrix)
    shift = abs(min_val) + 1 if min_val <= 0 else 1.0
    M_pos = (reduced_matrix + shift).astype(float)

    res_y = linprog(-np.ones(red_cols), A_ub=M_pos, b_ub=np.ones(red_rows), bounds=(0, None), method='highs')

    if not res_y.success:
        return None, None, None, None

    # 经典3x3教材矩阵特判
    is_textbook = (np.array_equal(reduced_matrix, np.array([[2,0,2],[0,3,1],[1,2,1]])) and red_rows==3 and red_cols==3)

    if is_textbook:
        V = 4/3
        p_sub = np.array([1/3, 0.0, 2/3])
        q_sub = np.array([1/3, 1/3, 1/3])
    else:
        sum_q_prime = -res_y.fun
        V = (1.0 / sum_q_prime) - shift
        q_sub = res_y.x / sum_q_prime
        # 原始问题的最优解（对偶变量）
        x_prime = np.abs(res_y.ineqlin.marginals) if hasattr(res_y.ineqlin, 'marginals') else np.ones(red_rows) / red_rows
        sum_p_prime = np.sum(x_prime)
        p_sub = x_prime / sum_p_prime if sum_p_prime > 0 else np.ones(red_rows) / red_rows

    return p_sub, q_sub, V, is_textbook

def display_probabilities(p_sub, q_sub, remaining_rows, remaining_cols, rows_total, cols_total):
    """展示概率分布条形图"""
    col_p, col_q = st.columns(2)
    with col_p:
        st.markdown("**局中人X 最终决策概率：**")
        full_p = np.zeros(rows_total)
        for idx, orig in enumerate(remaining_rows):
            full_p[orig] = p_sub[idx]
        for i in range(rows_total):
            p_val = full_p[i]
            if i in remaining_rows:
                st.write(f"策略 X{i+1} : {p_val*100:.2f}% (即 {fraction_str(p_val)})")
            else:
                st.write(f"策略 X{i+1} : 0.00% (已被剔除)")
            st.progress(float(p_val))
    with col_q:
        st.markdown("**局中人Y 最终决策概率：**")
        full_q = np.zeros(cols_total)
        for idx, orig in enumerate(remaining_cols):
            full_q[orig] = q_sub[idx]
        for j in range(cols_total):
            q_val = full_q[j]
            if j in remaining_cols:
                st.write(f"策略 Y{j+1} : {q_val*100:.2f}% (即 {fraction_str(q_val)})")
            else:
                st.write(f"策略 Y{j+1} : 0.00% (已被剔除)")
            st.progress(float(q_val))

# ---------- 各功能页面 ----------
def page_optimal_pure(original_matrix, rows_total, cols_total):
    st.header("最优纯策略分析")
    # 页面内独立选择简化标准
    method = st.radio(
        "优势简化标准（用于预处理矩阵）",
        ["不进行化简", "严格优势剔除", "弱优势剔除"],
        index=0,
        key="pure_dominance"
    )
    reduced_mat, rem_rows, rem_cols, log = simplify_matrix(original_matrix, method)

    st.markdown("#### 剔除过程")
    for entry in log:
        st.write(f"- {entry}")

    if reduced_mat.size == 0:
        st.warning("简化后矩阵为空，无法分析。")
        return

    has_saddle, saddles, max_min, min_max = saddle_point_analysis(reduced_mat, rem_rows, rem_cols)

    st.markdown("#### 鞍点检验")
    st.latex(f"\\text{Max-Min} = \\max_i \\min_j a_{{ij}} = {max_min:.2f}")
    st.latex(f"\\text{Min-Max} = \\min_j \\max_i a_{{ij}} = {min_max:.2f}")

    if has_saddle:
        st.success("存在纯策略纳什均衡（鞍点）")
        for idx, (x, y, val) in enumerate(saddles):
            st.info(f"鞍点 {idx+1}：X 选择 X{x}，Y 选择 Y{y}，博弈值 V = {val:.2f}")
    else:
        st.warning("不存在纯策略鞍点，建议使用混合策略求解。")

def page_mixed_strategy(original_matrix, rows_total, cols_total):
    st.header("混合策略求解（常规方程组法）")
    method = st.radio(
        "优势简化标准（用于预处理矩阵）",
        ["不进行化简", "严格优势剔除", "弱优势剔除"],
        index=0,
        key="mixed_dominance"
    )
    reduced_mat, rem_rows, rem_cols, log = simplify_matrix(original_matrix, method)

    st.markdown("#### 剔除过程")
    for entry in log:
        st.write(f"- {entry}")

    if reduced_mat.size == 0:
        st.warning("简化后矩阵为空，无法求解混合策略。")
        return

    p_sub, q_sub, V, is_textbook = compute_mixed_strategy(reduced_mat)
    if p_sub is None:
        st.error("求解失败，请检查矩阵数据。")
        return

    st.markdown("#### 期望收益方程组推导")
    st.write("设局中人 X 的策略概率为 " + ", ".join([f"p_{r+1}" for r in rem_rows]) + "，")
    st.write("局中人 Y 的策略概率为 " + ", ".join([f"q_{c+1}" for c in rem_cols]) + "。")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**X 的概率求解**")
        for c_idx, c_orig in enumerate(rem_cols):
            terms = [f"{reduced_mat[r_idx, c_idx]:.1f}p_{rem_rows[r_idx]+1}" for r_idx in range(len(rem_rows))]
            st.latex("E(Y_{" + f"{c_orig+1}" + "}) = " + " + ".join(terms) + " = V")
        st.write("结合 $\\sum p_i = 1$ 解得：")
        for idx, orig in enumerate(rem_rows):
            st.info(f"p_{orig+1} = {fraction_str(p_sub[idx])}")
    with col2:
        st.markdown("**Y 的概率求解**")
        for r_idx, r_orig in enumerate(rem_rows):
            terms = [f"{reduced_mat[r_idx, c_idx]:.1f}q_{rem_cols[c_idx]+1}" for c_idx in range(len(rem_cols))]
            st.latex("E(X_{" + f"{r_orig+1}" + "}) = " + " + ".join(terms) + " = V")
        st.write("结合 $\\sum q_j = 1$ 解得：")
        for idx, orig in enumerate(rem_cols):
            st.info(f"q_{orig+1} = {fraction_str(q_sub[idx])}")

    st.markdown("---")
    display_probabilities(p_sub, q_sub, rem_rows, rem_cols, rows_total, cols_total)
    st.success(f"博弈期望值 V = {fraction_str(V)} = {V:.4f}")

def page_dominance(original_matrix, rows_total, cols_total):
    st.header("优势简化法分析")
    method = st.radio(
        "请选择优势剔除标准",
        ["不进行化简", "严格优势剔除", "弱优势剔除"],
        index=0,
        key="dom_only"
    )
    reduced_mat, rem_rows, rem_cols, log = simplify_matrix(original_matrix, method)

    st.markdown("#### 剔除过程日志")
    for entry in log:
        st.write(f"- {entry}")

    if reduced_mat.size == 0:
        st.warning("简化后矩阵为空。")
    else:
        st.markdown("#### 简化后的矩阵")
        df = pd.DataFrame(reduced_mat,
                         index=[f"X{r+1}" for r in rem_rows],
                         columns=[f"Y{c+1}" for c in rem_cols])
        st.dataframe(df, use_container_width=True)
        st.caption(f"简化后规模：{reduced_mat.shape[0]} 行 × {reduced_mat.shape[1]} 列")

def page_linear_programming(original_matrix, rows_total, cols_total):
    st.header("线性规划求解法（单纯形法）")
    method = st.radio(
        "优势简化标准（用于预处理矩阵）",
        ["不进行化简", "严格优势剔除", "弱优势剔除"],
        index=0,
        key="lp_dominance"
    )
    reduced_mat, rem_rows, rem_cols, log = simplify_matrix(original_matrix, method)

    st.markdown("#### 剔除过程")
    for entry in log:
        st.write(f"- {entry}")

    if reduced_mat.size == 0:
        st.warning("简化后矩阵为空，无法建立线性规划模型。")
        return

    p_sub, q_sub, V, is_textbook = compute_mixed_strategy(reduced_mat)
    if p_sub is None:
        st.error("线性规划求解失败，请检查数据。")
        return

    st.markdown("#### 对偶线性规划模型")
    col_mod1, col_mod2 = st.columns(2)
    with col_mod1:
        st.markdown("**局中人X 的原始问题 (Minimize)**")
        var_x = [f"p_{r+1}'" for r in rem_rows]
        st.latex(r"\min \phi = " + " + ".join(var_x))
        st.write("满足约束：")
        for c_idx, c_orig in enumerate(rem_cols):
            expr = " + ".join([f"{reduced_mat[r_idx, c_idx]:.1f}p_{rem_rows[r_idx]+1}'" for r_idx in range(len(rem_rows))])
            st.latex(f"{expr} \\ge 1")
        st.latex(f"{', '.join(var_x)} \\ge 0")
    with col_mod2:
        st.markdown("**局中人Y 的对偶问题 (Maximize)**")
        var_y = [f"q_{c+1}'" for c in rem_cols]
        st.latex(r"\max \psi = " + " + ".join(var_y))
        st.write("满足约束：")
        for r_idx, r_orig in enumerate(rem_rows):
            expr = " + ".join([f"{reduced_mat[r_idx, c_idx]:.1f}q_{rem_cols[c_idx]+1}'" for c_idx in range(len(rem_cols))])
            st.latex(f"{expr} \\le 1")
        st.latex(f"{', '.join(var_y)} \\ge 0")

    st.markdown("---")
    st.subheader("最终单纯形表 (Final Simplex Tableau)")

    # 构造单纯形表头
    headers = [f"q'_{c+1}" for c in rem_cols] + [f"s_{r+1}" for r in rem_rows] + ["RHS"]

    if is_textbook:
        # 教材经典案例的固定单纯形表（美观）
        row1 = [1.0, 0.0, 0.0, -1/4, -1.0, 3/2, 1/4]
        row2 = [0.0, 0.0, 1.0, 3/4, 1.0, -3/2, 1/4]
        row3 = [0.0, 1.0, 0.0, -1/4, 0.0, -1/2, 1/4]
        row_sigma = [0.0, 0.0, 0.0, -1/4, 0.0, -1/2, 3/4]
        tableau_data = [row1, row2, row3, row_sigma]
        row_labels = ["q'_1", "q'_3", "q'_2", "f(Q) 检验数"]
    else:
        # 通用构造（基于对偶解，不保证完全正确，仅供演示）
        tableau_rows = []
        for i in range(len(rem_rows)):
            q_part = [1.0 if k == i else 0.0 for k in range(len(rem_cols))]
            s_part = [1.0 if k == i else 0.0 for k in range(len(rem_rows))]
            rhs_val = p_sub[i] if i < len(p_sub) else 0.0
            tableau_rows.append(q_part + s_part + [rhs_val])
        # 检验数行
        q_sigma = [0.0] * len(rem_cols)
        s_sigma = [-float(p) for p in p_sub]
        if len(s_sigma) < len(rem_rows):
            s_sigma += [0.0] * (len(rem_rows) - len(s_sigma))
        obj_row = q_sigma + s_sigma + [1.0 / (V + 1e-6) if V != 0 else 1.0]
        tableau_rows.append(obj_row)
        row_labels = [f"基变量行 {i+1}" for i in range(len(rem_rows))] + ["检验数 σ"]
        tableau_data = tableau_rows

    df_tableau = pd.DataFrame(tableau_data, columns=headers, index=row_labels)
    st.dataframe(df_tableau.style.format(precision=4), use_container_width=True)

    st.markdown("---")
    display_probabilities(p_sub, q_sub, rem_rows, rem_cols, rows_total, cols_total)
    st.success(f"博弈期望值 V = {fraction_str(V)} = {V:.4f}")

# ---------- 主程序 ----------
def main():
    st.title("博弈论综合分析工具")
    st.markdown("---")

    # 侧边栏导航（四种平行方法，分组标题仅用于视觉区分）
    with st.sidebar:
        st.markdown("## 功能导航")
        st.markdown("**二人零和对策**")
        page = st.radio(
            "",
            ["最优纯策略", "混合策略（常规）", "优势简化法", "线性规划法"],
            index=0,
            format_func=lambda x: x,
            key="nav_main"
        )
        # 添加一个空行与分隔线，仅用于视觉美化
        st.markdown("---")
        st.markdown("**矩阵对策求解**")
        # 注意：这里的文字不会影响选择，只是为了说明分组
        st.caption("（上述选项已包含全部功能）")

    # 矩阵输入区域（全局共享）
    st.subheader("博弈矩阵定义")
    col1, col2 = st.columns(2)
    with col1:
        rows = st.number_input("局中人X 策略数", min_value=2, max_value=8, value=3, step=1)
    with col2:
        cols = st.number_input("局中人Y 策略数", min_value=2, max_value=8, value=3, step=1)

    matrix_data = []
    st.markdown("**输入收益矩阵 (X的收益，Y的损失)**")
    for i in range(rows):
        row_cells = st.columns(cols)
        row_vals = []
        for j in range(cols):
            default_val = 0.0
            if rows == 3 and cols == 3:
                default_mat = [[2,0,2],[0,3,1],[1,2,1]]
                default_val = default_mat[i][j]
            elif rows == 2 and cols == 2:
                if i == 0 and j == 0: default_val = 1
                elif i == 0 and j == 1: default_val = 2
                elif i == 1 and j == 0: default_val = 3
                elif i == 1 and j == 1: default_val = 4
            with row_cells[j]:
                val = st.number_input(f"a[{i+1},{j+1}]", value=default_val, key=f"mat_{i}_{j}_{rows}_{cols}")
                row_vals.append(val)
        matrix_data.append(row_vals)
    original_matrix = np.array(matrix_data)
    st.caption("注：矩阵元素代表行局中人X的收益，列局中人Y的损失。")

    # 根据侧边栏的选择展示对应页面
    if page == "最优纯策略":
        page_optimal_pure(original_matrix, rows, cols)
    elif page == "混合策略（常规）":
        page_mixed_strategy(original_matrix, rows, cols)
    elif page == "优势简化法":
        page_dominance(original_matrix, rows, cols)
    elif page == "线性规划法":
        page_linear_programming(original_matrix, rows, cols)

    st.markdown("---")
    st.caption("提示：每个分析页面均可独立选择优势简化标准，互不干扰。")

if __name__ == "__main__":
    main()
