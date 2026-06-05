def page_dominance(matrix):
    st.header("优势简化法分析")
    reduced_mat, rem_rows, rem_cols, log = weak_dominance_simplify(matrix)

    st.markdown("#### 弱优势剔除过程日志")
    for entry in log:
        st.write(f"- {entry}")

    if reduced_mat.size == 0:
        st.warning("简化后矩阵为空，无法进行后续分析。")
        return

    st.markdown("#### 简化后的矩阵")
    df = pd.DataFrame(reduced_mat,
                     index=[f"X{r+1}" for r in rem_rows],
                     columns=[f"Y{c+1}" for c in rem_cols])
    st.dataframe(df, use_container_width=True)
    st.caption(f"简化后规模：{reduced_mat.shape[0]} 行 × {reduced_mat.shape[1]} 列")

    # ========== 新增：对简化矩阵进行鞍点检验 ==========
    st.markdown("---")
    st.subheader("简化矩阵的鞍点检验")
    # 注意：saddle_point_analysis 返回的鞍点坐标是相对于简化矩阵的行列索引
    has_saddle, saddles, max_min, min_max = saddle_point_analysis(reduced_mat)
    st.latex(f"\\text{{Max-Min}} = \\max_i \\min_j a_{{ij}} = {max_min:.2f}")
    st.latex(f"\\text{{Min-Max}} = \\min_j \\max_i a_{{ij}} = {min_max:.2f}")

    if has_saddle:
        st.success("简化后的矩阵存在纯策略纳什均衡（鞍点）")
        # 将鞍点坐标映射回原始策略编号
        for idx, (r, c, val) in enumerate(saddles):
            orig_r = rem_rows[r-1] + 1   # r是1-indexed
            orig_c = rem_cols[c-1] + 1
            st.info(f"鞍点 {idx+1}：X 选择 X{orig_r}，Y 选择 Y{orig_c}，博弈值 V = {val:.2f}")
    else:
        st.warning("简化后的矩阵不存在纯策略鞍点，转入混合策略求解。")

    # ========== 新增：对简化矩阵进行混合策略求解 ==========
    st.markdown("---")
    st.subheader("简化矩阵的混合策略求解")
    p_sub, q_sub, V, is_textbook = compute_mixed_strategy(reduced_mat)
    if p_sub is None:
        st.error("混合策略求解失败，请检查简化矩阵数据。")
        return

    # 动态生成字母表示概率（基于简化后的策略数量）
    red_rows, red_cols = reduced_mat.shape
    x_vars = get_letter_vars(red_rows)
    y_vars = get_letter_vars(red_cols)

    st.markdown("#### 期望收益方程组推导")
    st.write(f"设局中人 X 采用简化后各策略的概率分别为 {', '.join(x_vars)}，")
    st.write(f"局中人 Y 采用简化后各策略的概率分别为 {', '.join(y_vars)}。")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**X 的概率求解**")
        for j in range(red_cols):
            terms = [f"{reduced_mat[i, j]:.1f}{x_vars[i]}" for i in range(red_rows)]
            st.latex(f"E(Y_{{{rem_cols[j]+1}}}) = " + " + ".join(terms) + " = V")
        st.write(f"结合 $\\sum " + " + ".join(x_vars) + " = 1$ 解得：")
        for i in range(red_rows):
            st.info(f"{x_vars[i]} = {fraction_str(p_sub[i])} (对应原始策略 X{rem_rows[i]+1})")
    with col2:
        st.markdown("**Y 的概率求解**")
        for i in range(red_rows):
            terms = [f"{reduced_mat[i, j]:.1f}{y_vars[j]}" for j in range(red_cols)]
            st.latex(f"E(X_{{{rem_rows[i]+1}}}) = " + " + ".join(terms) + " = V")
        st.write(f"结合 $\\sum " + " + ".join(y_vars) + " = 1$ 解得：")
        for j in range(red_cols):
            st.info(f"{y_vars[j]} = {fraction_str(q_sub[j])} (对应原始策略 Y{rem_cols[j]+1})")

    st.markdown("---")
    # 展示概率条形图（基于简化矩阵的概率，并映射到原始策略编号）
    col_p, col_q = st.columns(2)
    with col_p:
        st.markdown("**局中人X 最终决策概率（简化后策略）**")
        for i in range(red_rows):
            orig = rem_rows[i]
            st.write(f"原始策略 X{orig+1} : {p_sub[i]*100:.2f}% (即 {fraction_str(p_sub[i])})")
            st.progress(float(p_sub[i]))
    with col_q:
        st.markdown("**局中人Y 最终决策概率（简化后策略）**")
        for j in range(red_cols):
            orig = rem_cols[j]
            st.write(f"原始策略 Y{orig+1} : {q_sub[j]*100:.2f}% (即 {fraction_str(q_sub[j])})")
            st.progress(float(q_sub[j]))

    st.success(f"基于简化矩阵的博弈期望值 V = {fraction_str(V)} = {V:.4f}")
