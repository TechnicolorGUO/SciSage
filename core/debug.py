import re


def aggregate_references(text):
    # 使用正则表达式匹配[数字]的模式
    pattern = r"\[\d+\]"
    matches = list(re.finditer(pattern, text))

    # 如果匹配结果少于2个，直接返回原文本
    if len(matches) < 2:
        return text

    # 创建一个集合存储最后一个出现的编号
    result = []
    last_seen = None

    for i, match in enumerate(matches):
        ref = match.group()
        print("ref:", ref, last_seen)
        # 仅保留最后一个编号
        if last_seen != ref:
            if i + 1 < len(matches) and matches[i + 1].group() == ref:
                continue
            last_seen = ref

            result.append((match.start(), match.end(), ref))
            print("result:", result)

    ## 构造结果字符串
    new_text = []
    last_pos = 0

    for start, end, ref in result:
        new_text.append(text[last_pos:start].replace(ref, ""))
        new_text.append(ref)
        last_pos = end

    new_text.append(text[last_pos:])
    return "".join(new_text)


# 测试内容
content = """
LangGraph is a graph-based framework that plays a crucial role in AI agent frameworks, particularly in enhancing machine translation, real-time data analysis, and decision-making [2]. Its significance lies in its ability to simplify the creation and management of agents and their workflows, enabling efficient state management, dynamic workflow construction, and robust memory checkpointing [2]. The importance of language-based graph processing in LangGraph cannot be overstated, as it allows agents to dynamically determine control flows, invoke tools, and assess the necessity of further actions, improving flexibility and efficiency [2]. Moreover, LangGraph's graph-structured workflows enable agents to execute complex tasks, adapt to new inputs, and provide real-time feedback, ensuring seamless decision-making and execution in distributed environments [2]. However, it's worth noting that LangGraph may have limitations, such as potential drawbacks or limitations in certain applications [0]. For instance, a study on the application of Spark Streaming real-time data analysis system and large language model intelligent agents demonstrates the potential of LangGraph to enhance multilingual translation accuracy and scalability [3]. Another study on intelligent Spark agents highlights the framework's ability to simplify machine learning processes by allowing users to visually design workflows, which are then converted into Spark-compatible code for high-performance execution [2]. Additionally, LangGraph has been successfully applied in various AI agent frameworks, such as Agent AI with LangGraph, which has been shown to enhance machine translation using large language models [4]. The framework enables agents to perform specific tasks, such as translating between particular languages, while maintaining modularity, scalability, and context retention [4]. Furthermore, LangGraph has been used in intelligent Spark agents to enhance machine learning workflows through scalability, visualization, and intelligent process optimization [2]. The framework automates data preprocessing, feature engineering, and model evaluation, while dynamically interacting with data through Spark SQL and DataFrame agents [2]. LangGraph has also been explored in the context of multi-agent systems, where it improves the efficiency of information transmission through graph architecture [0]. Moreover, LangGraph has been used to develop an advanced RAG system based on graph technology, which efficiently searches and utilizes information to generate more accurate and enhanced responses [1]. In summary, LangGraph is a crucial component of AI agent frameworks, enabling efficient state management, dynamic workflow construction, and robust memory checkpointing [2]. Its language-based graph processing capabilities make it an essential tool for enhancing machine translation, real-time data analysis, and decision-making [2]. While it may have limitations, the supporting scholarly articles demonstrate the potential of LangGraph to improve various AI applications, making it a valuable resource for practical application [0][1][2][3][4].
"""
# 运行函数
result = aggregate_references(content)
print(result)
