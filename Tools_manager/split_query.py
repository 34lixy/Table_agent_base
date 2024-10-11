from Model_manager.API_service import CustomLLM


def split_query(query, data_str):
    """
    根据数据data_str, 将用户的问题query拆分成6个细分的子问题。
    """
    prompt_template = '''请理解关键列的指标、表格名称、表头的指标，从生成报告的角度产生4-5个可以分析子问题，第一个子问题需要概括性描述整体数据，如整体趋势、地区差异、指标基本情况等，最后一个子问题总结上述所有问题并形成结论，并针对性的提供建议。中间的2-3子问题，每个子问题从下列角度中挑选一个产生：
    指标随时间的变化趋势；指标的月度季节性变化；指标的周期性变化；指标同比或环比增幅和下降情况；指标异常值数据（前几名或后几名）；各部分指标占总指标的比例；同一指标在不同时间的变化；同一指标在不同地区的变化；同一指标在不同类别或领域的变化；不同指标在同一时间地区的比较。
    注意不要照搬已有角度，应契合已有数据内容，确保已有数据能回答该问题。仅输出问题。
    '''
    llm = CustomLLM()
    input_str = prompt_template.format(query)
    response = llm.chat(input_str, '数据如下：\n' + data_str)
    return response
