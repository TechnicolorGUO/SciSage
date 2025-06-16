# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================

RESPONSE_START_DELIMITER = "[Response_Start]"
RESPONSE_END_DELIMITER = "[Response_End]"
REFERENCES_HEADER = "References:"
REVISED_ANSWER_HEADER = "Here is the revised answer:\n\n"


chat_system = """You are an expert AI assistant specialized in scientific writing and research. You help researchers create high-quality academic content based on recent scientific literature.

Your capabilities include:
- Generating comprehensive responses to scientific questions using provided research papers
- Writing well-structured academic sections with proper citations
- Providing constructive feedback on scientific writing
- Incorporating feedback to improve academic content
- Adding proper citations and attributions to scientific claims
- Analyzing and reformatting content for academic standards

Guidelines:
- Always base your responses on the provided scientific literature
- Use proper citation format [X] where X is the reference number
- Maintain academic writing style and clarity
- Be precise and factual in your statements
- When asked for feedback, provide specific and actionable suggestions
- Follow the exact format requirements specified in prompts
- Preserve important scientific details and evidence"""


# Task Instructions
task_instructions = {
    "claim_no_context": (
        "Given a scientific claim, answer if the scientific claim is factually correct (true) or not (false). For each scientific claim provided, simply state whether it is true or false. If the statement is supported by the paragraph, answer true; otherwise answer false. You don't need to provide any explanation, just the label.",
        "\nClaim: ",
    ),
    "claim_gold": (
        "Given a scientific claim and a gold paragraph that may support or contradict with the claim, answer if the scientific claim is factually correct or not. For each scientific claim provided, simply state whether it is true or false. If the statement is supported by the paragraph, answer true; otherwise answer false. You don't need to provide any explanation, just the label.",
        "\nClaim: ",
    ),
    "claim_full": (
        "Given a scientific claim and a set of relevant paragraphs, that may support or contradict with the claim, answer if the scientific claim is factually correct or not. For each scientific claim provided, simply state whether it is true or false. If the statement is supported by the paragraph, answer true; otherwise answer false. You don't need to provide any explanation, just the label. You also need to provide the citation numbers that support your answer. Your citation is presented as [i], where i corresponds to the number in the 'References: '.",
        "\nClaim: ",
    ),
    "boolean_question_no_context": (
        "Given a question related to scientific literature, answer yes or no. Simply state whether it is yes or no. You don't need to provide any explanation, just the label.",
        "\nQuestion: ",
    ),
    "boolean_question_gold": (
        "Given a question related to scientific literature and a gold paragraph that provides sufficient information to answer the question, answer yes or no. Simply state whether it is yes or no.",
        "\nQuestion:",
    ),
    "boolean_question_full": (
        "Given a question related to scientific literature and a set of reference passages that may provide sufficient information to answer the question, answer yes or no. Simply state whether it is yes or no. You don't need to provide any explanation, just the label. You also need to provide the citation numbers that support your answer. Your citation is presented as [i], where i corresponds to the number in the 'References: '.",
        "\nQuestion: ",
    ),
}

demonstrations = {
    "claim_no_context": "Your answer must be marked by special tokens, [Response_Start] and [Response_End]. For example, the input and output looks like this:\nClaim: 1 in 5 million in UK have abnormal PrP positivity.\n[Response_Start]false[Response_End]\nNow please verify the following claim.",
    "claim_gold": "Your answer must be marked by special tokens, [Response_Start] and [Response_End]. For example, the input and output looks like this: \nReferences: \n[0] Title: Prevalent abnormal prion protein in human appendixes after bovine spongiform encephalopathy epizootic: large scale survey Text: OBJECTIVES To carry out a further survey of archived appendix samples to understand better the differences between existing estimates of the prevalence of subclinical infection with prions after the bovine spongiform encephalopathy epizootic and to see whether a broader birth cohort was affected, and to understand better the implications for the management of blood and blood products and for the handling of surgical instruments. DESIGN Irreversibly unlinked and anonymised large scale survey of archived appendix samples. SETTING Archived appendix samples from the pathology departments of 41 UK hospitals participating in the earlier survey, and additional hospitals in regions with lower levels of participation in that survey. SAMPLE 32,441 archived appendix samples fixed in formalin and embedded in paraffin and tested for the presence of abnormal prion protein (PrP). RESULTS Of the 32,441 appendix samples 16 were positive for abnormal PrP, indicating an overall prevalence of 493 per million population (95% confidence interval 282 to 801 per million). The prevalence in those born in 1941-60 (733 per million, 269 to 1596 per million) did not differ significantly from those born between 1961 and 1985 (412 per million, 198 to 758 per million) and was similar in both sexes and across the three broad geographical areas sampled. Genetic testing of the positive specimens for the genotype at PRNP codon 129 revealed a high proportion that were valine homozygous compared with the frequency in the normal population, and in stark contrast with confirmed clinical cases of vCJD, all of which were methionine homozygous at PRNP codon 129. CONCLUSIONS This study corroborates previous studies and suggests a high prevalence of infection with abnormal PrP, indicating vCJD carrier status in the population compared with the 177 vCJD cases to date. These findings have important implications for the management of blood and blood products and for the handling of surgical instruments.\nClaim: 1 in 5 million in UK have abnormal PrP positivity. \n[Response_Start]false[Response_End]\nNow please verify the following claim.\n",
    "claim_full": """
    Your answer must be marked by special tokens, [Response_Start] and [Response_End]. For example, the input and output looks like this:
    References:
    [0] Title: MLQA: Evaluating Cross-lingual Extractive Question Answering Text: Question answering (QA) models have shown rapid progress enabled by the availability of large, high-quality benchmark datasets. Such annotated datasets are difficult and costly to collect, and rarely exist in languages other than English, making building QA systems that work well in other languages challenging. In order to develop such systems, it is crucial to invest in high quality multilingual evaluation benchmarks to measure progress. We present MLQA, a multi-way aligned extractive QA evaluation benchmark intended to spur research in this area. MLQA contains QA instances in 7 languages, English, Arabic, German, Spanish, Hindi, Vietnamese and Simplified Chinese. MLQA has over 12K instances in English and 5K in each other language, with each instance parallel between 4 languages on average. We evaluate state-of-the-art cross-lingual models and machine-translation-based baselines on MLQA. In all cases, transfer results are shown to be significantly behind training-language performance.
    [1] Title: XOR QA: Cross-lingual Open-Retrieval Question Answering Text: Multilingual question answering tasks typically assume that answers exist in the same language as the question. Yet in practice, many languages face both information scarcity—where languages have few reference articles—and information asymmetry—where questions reference concepts from other cultures. This work extends open-retrieval question answering to a cross-lingual setting enabling questions from one language to be answered via answer content from another language. We construct a large-scale dataset built on 40K information-seeking questions across 7 diverse non-English languages that TyDi QA could not find same-language answers for. Based on this dataset, we introduce a task framework, called Cross-lingual Open-Retrieval Question Answering (XOR QA), that consists of three new tasks involving cross-lingual document retrieval from multilingual and English resources. We establish baselines with state-of-the-art machine translation systems and cross-lingual pretrained models. Experimental results suggest that XOR QA is a challenging task that will facilitate the development of novel techniques for multilingual question answering.
    [2] Title: Unsupervised Cross-lingual Representation Learning at Scale Text: This paper shows that pretraining multilingual language models at scale leads to significant performance gains for a wide range of cross-lingual transfer tasks. We train a Transformer-based masked language model on one hundred languages, using more than two terabytes of filtered CommonCrawl data. Our model, dubbed XLM-R, significantly outperforms multilingual BERT (mBERT) on a variety of cross-lingual benchmarks, including +14.6% average accuracy on XNLI, +13% average F1 score on MLQA, and +2.4% F1 score on NER. XLM-R performs particularly well on low-resource languages, improving 15.7% in XNLI accuracy for Swahili and 11.4% for Urdu over previous XLM models. We also present a detailed empirical analysis of the key factors that are required to achieve these gains, including the trade-offs between (1) positive transfer and capacity dilution and (2) the performance of high and low resource languages at scale. Finally, we show, for the first time, the possibility of multilingual modeling without sacrificing per-language performance; XLM-R is very competitive with strong monolingual models on the GLUE and XNLI benchmarks. We will make our code, data and models publicly available.
    Claim: The XOR QA dataset covers eight languages.
    [Response_Start]false [1][Response_End]
    Now please verify the following claim.\n
    """,
    "boolean_question_no_context": "Your answer must be marked by special tokens, [Response_Start] and [Response_End]. For example, the input and output looks like this:\nQuestion: Did Chile's traffic law reform push police enforcement?\n[Response_Start]yes[Response_End]\nNow answer the following question.",
    "boolean_question_gold": "Your answer must be marked by special tokens, [Response_Start] and [Response_End]. For example, the input and output looks like this: \nReferences: \n[0] The objective of the current study is to determine to what extent the reduction of Chile's traffic fatalities and injuries during 2000-2012 was related to the police traffic enforcement increment registered after the introduction of its 2005 traffic law reform. A unique dataset with assembled information from public institutions and analyses based on ordinary least square and robust random effects models was carried out. Dependent variables were traffic fatality and severe injury rates per population and vehicle fleet. Independent variables were: (1) presence of new national traffic law; (2) police officers per population; (3) number of traffic tickets per police officer; and (4) interaction effect of number of traffic tickets per police officer with traffic law reform. Oil prices, alcohol consumption, proportion of male population 15-24 years old, unemployment, road infrastructure investment, years' effects and regions' effects represented control variables. Empirical estimates from instrumental variables suggest that the enactment of the traffic law reform in interaction with number of traffic tickets per police officer is significantly associated with a decrease of 8% in traffic fatalities and 7% in severe injuries. Piecewise regression model results for the 2007-2012 period suggest that police traffic enforcement reduced traffic fatalities by 59% and severe injuries by 37%. \nQuestion: Did Chile's traffic law reform push police enforcement?\n[Response_Start]yes[Response_End]\nNow answer the following question. ",
    "boolean_question_full": """
    Your answer must be marked by special tokens, [Response_Start] and [Response_End]. For example, the input and output looks like this:
    References:
    [0] The gap between evidence-based treatments and routine care has been well established. Findings from the Sequenced Treatments Alternatives to Relieve Depression (STAR*D) emphasized the importance of measurement-based care for the treatment of depression as a key ingredient for achieving response and remission; yet measurement-based care approaches are not commonly used in clinical practice. The Nine-Item Patient Health Questionnaire (PHQ-9) for monitoring depression severity was introduced in 19 diverse psychiatric practices. During the one-year course of the project the helpfulness and feasibility of implementation of PHQ-9 in these psychiatric practices were studied. The project was modeled after the Institute for Healthcare Improvement Breakthrough Series. Two of the 19 practices dropped out during the course of the project. By the conclusion of the study, all remaining 17 practices had adopted PHQ-9 as a routine part of depression care in their practice. On the basis of responses from 17 psychiatrists from those practices, PHQ-9 scores influenced clinical decision making for 93% of 6,096 patient contacts. With the additional information gained from the PHQ-9 score, one or more treatment changes occurred during 40% of these clinical contacts. Changing the dosage of antidepressant medication and adding another medication were the most common treatment changes recorded by psychiatrists, followed by starting or increasing psychotherapy and by switching or initiating antidepressants. In 3% of the patient contacts, using the PHQ-9 led to additional suicide risk assessment.
    [1] To compare maternal and neonatal outcomes among grandmultiparous women to those of multiparous women 30 years or older. A database of the vast majority of maternal and newborn hospital discharge records linked to birth/death certificates was queried to obtain information on all multiparous women with a singleton delivery in the state of California from January 1, 1997 through December 31, 1998. Maternal and neonatal pregnancy outcomes of grandmultiparous women were compared to multiparous women who were 30 years or older at the time of their last birth. The study population included 25,512 grandmultiparous and 265,060 multiparous women 30 years or older as controls. Grandmultiparous women were predominantly Hispanic (56%). After controlling for potential confounding factors, grandmultiparous women were at significantly higher risk for abruptio placentae (odds ratio OR: 1.3; 95% confidence intervals CI: 1.2-1.5), preterm delivery (OR: 1.3; 95% CI: 1.2-1.4), fetal macrosomia (OR: 1.5; 95% CI: 1.4-1.6), neonatal death (OR: 1.5; 95% CI: 1.3-1.8), postpartum hemorrhage (OR: 1.2; 95% CI: 1.1-1.3) and blood transfusion (OR: 1.5; 95% CI: 1.3-1.8).', 'long_answer': 'Grandmultiparous women had increased maternal and neonatal morbidity, and neonatal mortality even after controlling for confounders, suggesting a need for closer observation than regular multiparous patients during labor and delivery.
    [2] The objective of the current study is to determine to what extent the reduction of Chile's traffic fatalities and injuries during 2000-2012 was related to the police traffic enforcement increment registered after the introduction of its 2005 traffic law reform. A unique dataset with assembled information from public institutions and analyses based on ordinary least square and robust random effects models was carried out. Dependent variables were traffic fatality and severe injury rates per population and vehicle fleet. Independent variables were: (1) presence of new national traffic law; (2) police officers per population; (3) number of traffic tickets per police officer; and (4) interaction effect of number of traffic tickets per police officer with traffic law reform. Oil prices, alcohol consumption, proportion of male population 15-24 years old, unemployment, road infrastructure investment, years' effects and regions' effects represented control variables. Empirical estimates from instrumental variables suggest that the enactment of the traffic law reform in interaction with number of traffic tickets per police officer is significantly associated with a decrease of 8% in traffic fatalities and 7% in severe injuries. Piecewise regression model results for the 2007-2012 period suggest that police traffic enforcement reduced traffic fatalities by 59% and severe injuries by 37%.
    Question: Did Chile's traffic law reform push police enforcement?
    [Response_Start]yes [2][Response_End]
    Now answer the following question.
    """,
}
# Examples
example_passages_rag = """
[0] Title: Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models Text: Roberts et al. (2020) shows that T5 (Raffel et al., 2020) can perform a new task formulation, closedbook QA. Concretely, T5 can produce answers to questions without access to any corpus at inference time, instead producing answers based on its model parameters, tuned to remember information digested in pretraining.
[1] Title: Reliable, Adaptable, and Attributable Language Models with Retrieval Text: Unlike parametric LMs—which use large-scale text data only during training—retrieval-augmented LMs leverage an external large-scale collection of documents (datastore) at inference by selecting relevant documents from the datastore (Asai et al., 2023a). Retrieval-augmented LMs can W1: largely reduce factual errors (Mallen et al., 2023), W2: provide better attributions (Gao et al., 2023a), W3: enabling flexible opt-in and out of sequences (Min et al., 2024).
[2] Title: Atlas: Few-shot Learning with Retrieval Augmented Language Models Text: In this work we present Atlas, a carefully designed and pre-trained retrieval augmented language model able to learn knowledge intensive tasks with very few training examples. We perform evaluations on a wide range of tasks, including MMLU, KILT and NaturalQuestions, and study the impact of the content of the document index, showing that it can easily be updated. Notably, Atlas reaches over 42% accuracy on Natural Questions using only 64 examples, outperforming a 540B parameters model by 3% despite having 50x fewer parameters.
[3] Title: Language Models are Few-Shot Learners Text: Similarly, GPT-3 achieves 64.3% accuracy on TriviaQA in the zero-shot setting, 68.0% in the one-shot setting, and 71.2% in the few-shot setting, the last of which is state-of-the-art relative to fine-tuned models operating in the same closed-book setting.
[4] Title: When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories Text:  On both datasets, LMs’ memorization (RQ1) is often limited to the popular factual knowledge and even GPT-3 davinci-003 fails to answer the majority of the long-tail questions. Moreover, on such questions, scaling up models does not significantly improve the performance. This also suggests that we can predict if LMs memorize certain knowledge based on the information presented in the input question only. We next investigate whether a semi-parametric approach that augments LMs with retrieved evidence can mitigate the low performance on questions about less popular entities (RQ2). Nonparametric memories largely improve performance on long-tail distributions across models.
[5] Title: Democratizing Large Language Models via Personalized Parameter-Efficient Fine-tuning Text: Personalization in large language models (LLMs) is increasingly important, aiming to align LLM’s interactions, content, and recommendations with individual user preferences. Recent advances in LLM personalization have spotlighted effective prompt design, by enriching user queries with non-parametric knowledge through behavior history retrieval and textual profiles. However, these approaches were limited due to a lack of model ownership, resulting in constrained customization and privacy issues. Moreover, they often failed to accurately capture user behavior patterns, especially in cases where user data were complex and dynamic. To address these shortcomings, we introduce One PEFT Per User (OPPU), which employs personalized parameter-efficient fine-tuning (PEFT) modules, to store user-specific behavior patterns and preferences.
[6] Title: RECOMP: Improving Retrieval-Augmented LMs with Context Compression and Selective Augmentation Text:  Retrieval-augmented language models (RALMs) (Khandelwal et al., 2019; Izacard et al., 2022; Lewis et al., 2020; Borgeaud et al., 2022) have shown impressive performance on knowledge-intensive tasks (Kwiatkowski et al., 2019; Petroni et al., 2021). Simply prepending retrieved documents to the input without updating the language models (LMs) (Shi et al., 2023b; Ram et al., 2023; Si et al., 2022) allows retrieval augmentation even for black-box LMs, but such approach comes with limitations. First, it increases computational costs as LMs now encode substantially more tokens. Second, even if we manage to adapt LMs to efficiently incorporate longer context (Beltagy et al., 2020; Zaheer et al., 2020), these models struggle to use all information in the context, frequently missing information placed in the middle (Liu et al., 2023). Third, prepending a large number of documents in-context can further confuse LMs with irrelevant information, degrading model performances (Mallen et al., 2022; Shi et al., 2023a).
"""
example_question_rag = (
    "How do language models leverage parametric and non-parametric knowledge?"
)
example_answer_rag = """
Language models leverage both parametric and non-parametric knowledge to perform various tasks.\n
Parametric knowledge refers to the information stored in the model's parameters, which are learned during training [0]. This type of knowledge allows language models to perform tasks such as closed-book question answering, where the model produces answers based on its internal knowledge without accessing any external corpus [0]. However, language models' memorization of parametric knowledge is often limited to popular factual knowledge, and even large models like GPT-3 may fail to answer the majority of long-tail questions [4].\n
On the other hand, non-parametric knowledge is retrieved from an external source, such as a large-scale collection of documents, during inference [1]. This type of knowledge is used in retrieval-augmented language models, which can reduce factual errors, provide better attributions, and enable flexible opt-in and out of sequences [1]. Retrieval-augmented language models have been shown to be effective in few-shot learning scenarios, where they can learn knowledge-intensive tasks with very few training examples [2]. For example, the Atlas model, a retrieval-augmented language model, can reach over 42% accuracy on Natural Questions using only 64 examples, outperforming a 540B parameters model by 3% despite having 50x fewer parameters [2]. Moreover, even without training, simply combining off-the-shelf LMs such as GPT3 with retrieval augmentation can significantly improve performance in long-tail and have been shown to mitigate the low performance on questions about less popular entities[4]. However, retrieval-augmented LMs have several limitations. Specifically, retrieval-augmented LMs can make inference much more inefficient due to increased context length [6].\n
"""
example_answer_rag_incorrect = """
Language models leverage both parametric and non-parametric knowledge to perform various tasks. Parametric knowledge refers to the information stored in the model's parameters, which are learned during training [0]. This type of knowledge allows language models to perform tasks such as closed-book question answering, where the model produces answers based on its internal knowledge without accessing any external corpus [0]. However, language models' memorization of parametric knowledge is often limited to popular factual knowledge, and even large models like GPT-4 often fail to answer the majority of long-tail questions [4].\n
On the other hand, non-parametric knowledge is retrieved from an external source, such as a large-scale collection of documents, during inference [1]. This type of knowledge is used in retrieval-augmented language models, which can reduce factual errors, provide better attributions, and enable flexible opt-in and out of sequences [1]. Retrieval-augmented language models have been shown to be effective in few-shot learning scenarios, where they can learn knowledge-intensive tasks with very few training examples [2]. For example, the Atlas model, a retrieval-augmented language model, can reach over 42% accuracy on Natural Questions using only 64 examples, outperforming a 540B parameters model by 3% despite having 50x fewer parameters [2]. Moreover, even without training, simply combining off-the-shelf LMs such as GPT3 with retrieval augmentation can significantly improve performance in long-tail and have been shown to mitigate the low performance on questions about less popular entities [4]. However, retrieval-augmented LMs have several limitations. Specifically, retrieval-augmented LMs can make inference much more inefficient due to increased context length [6].\n
"""

example_feedback = """
Feedback: Only concrete examples used in the answer are QA results. We should include more results from non QA tasks. Question: What tasks retrieval-augmented LMs have been applied to?\n
Feedback: Only one limitation discussed in the answer is efficiency. Question: What are the disadvantages of retrieval-augmented LMs?\n
Feedback: The original answer can be improved by adding more logical structure e.g., grouping similar discussions together and add paragraph headers.\n
"""

example_question_peft = "Discuss various parameter-efficient fine-tuning (PEFT) techniques for large language models, highlighting their strengths and weaknesses."
example_passages_peft = """
[0] Title: Empirical Analysis of the Strengths and Weaknesses of PEFT Techniques for LLMs Text: As foundation models continue to exponentially scale in size, efficient methods of adaptation become increasingly critical. Parameter-efficient fine-tuning (PEFT), a recent class of techniques that require only modifying a small percentage of the model parameters, is currently the most popular method for adapting large language models (LLMs). Several PEFT techniques have recently been proposed with varying tradeoffs. We provide a comprehensive and uniform benchmark of various PEFT techniques across a representative LLM, the FLAN-T5 model, and evaluate model performance across different data scales of classification and generation datasets. Based on this, we provide a framework for choosing the optimal fine-tuning techniques given the task type and data availability. Contrary to popular belief, we also empirically prove that PEFT techniques converge slower than full tuning in low data scenarios, and posit the amount of data required for PEFT methods to both perform well and converge efficiently.\n
[1] Title: Prefix-Tuning: Optimizing Continuous Prompts for Generation Text: In this paper, we propose prefix-tuning, a lightweight alternative to fine-tuning for natural language generation tasks, which keeps language model parameters frozen, but optimizes a small continuous task-specific vector (called the prefix). Prefix-tuning draws inspiration from prompting, allowing subsequent tokens to attend to this prefix as if it were "virtual tokens". We apply prefix-tuning to GPT-2 for table-to-text generation and to BART for summarization. We find that by learning only 0.1\% of the parameters, prefix-tuning obtains comparable performance in the full data setting, outperforms fine-tuning in low-data settings, and extrapolates better to examples with topics unseen during training.\n
[2] Title: Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models: A Survey Text: This paper aims to provide a comprehensive and systematic study of PEFT methods in the vision domain, particularly focusing on transformer-based pre-trained models ranging from the year 2019 to the year 2023. As shown in Fig. 1, existing visual PEFT methods could be categorized into addition-based tuning, partial-based tuning, and unified-based tuning. In section 2, we will define the problem of PEFT, introduce popular backbones, and discuss pre-training methods. In section 3, a detailed taxonomy and in-depth analysis of the PEFT methods will be presented. The real-world applications of PEFT will be introduced in section 4. Finally, in section 5, we will point out future research challenges.\n
[3] Title: Towards a Unified View of Parameter-Efficient Transfer Learning Text: To mitigate this issue, a few lightweight alternatives have been proposed to update only a small number of extra parameters while keeping most pretrained parameters frozen. For example, adapter tuning (Houlsby et al., 2019) inserts small neural modules called adapters to each layer of the pretrained network and only the adapters are trained at fine-tuning time. Inspired by the success of prompting methods that control PLMs through textual prompts (Brown et al., 2020; Liu et al., 2021a), prefix tuning (Li & Liang, 2021) and prompt tuning (Lester et al., 2021) prepend an additional l tunable prefix tokens to the input or hidden layers and only train these soft prompts when fine-tuning on downstream tasks.\n
[4] Title: I-Tuning: Tuning Frozen Language Models with Image for Lightweight Image Captioning  Text: We design an I-Tuning module to connect the pre-trained vision encoder (i.e., CLIP-ViT [7]) and the language decoder (i.e., GPT2 [8]). To align between the language and vision modals, it serves as a cross-modal filter that automatically picks the visual information from the output of the vision encoder and adjusts the output hidden states of the language decoder. During training, we only update the newly introduced parameters in the I-Tuning module, and the parameters of the two pre-trained models are frozen.\n
"""
example_rating_peft = """
[Response_Start][0] Rating: 3 Explanation: This paragraph discusses a high-level overview and goal of parameter efficient tuning but does not mention any particular methods of parameter efficient tuning and thus may not be super helpful. This could still be useful to discuss general advantages of PEFT.
[1] Rating: 5 Explanation: This paragraph introduces Prefix Tuning, one of the most representative methods in parameter efficient tuning and includes their core empirical results.
[2] Rating: 3 Explanation: While this paragraph provides a taxonomy of parameter efficient tuning and analysis, it does not provide any details of individual methods. Moreover, this paper's main focus is PEFT for vision models, while the original question asks about parameter efficient tuning for large language models.
[3] Rating: 4 Explanation: This paragraph briefly introduces multiple parameter efficient tuning methods such as adapter tuning, prefix tuning and prompt tuning. While they do not directly discuss their advantages or disadvantages or more detail about prefix or prompt tuning, still this paragraph gives a useful overview of this area.
[4] Rating: 1 Explanation: This paragraph introduces a new parameter efficient tuning method to connect a vision encoder and language encoder to make their representations aligned. The question asks about representative approaches of parameter efficient tuning for large language models, and this paragraph topic is substantially different from the question.[Response_End]\n
"""


## NOTE: Feedback
instruction_feedback = """
You are an expert scientific reviewer. Given an answer to a scientific question based on recent literature, provide constructive feedback to improve the response quality.

**Feedback Guidelines:**
1. **Content Quality**: Assess completeness, accuracy, and depth of coverage
2. **Evidence Support**: Evaluate citation quality and supporting evidence
3. **Structure & Clarity**: Review organization, flow, and readability
4. **Scope & Balance**: Check for missing perspectives or important details

**Feedback Format:**
- Start each point with "Feedback: "
- For content gaps requiring additional literature, add "Question: " with a self-contained search query
- Prioritize the most critical improvements first
- Be specific and actionable

**Example:**
Question: {example_question}
Answer: {example_answer}
[Response_Start]{example_feedback}[Response_End]

Now provide feedback for the following:
"""

# Improved example feedback with more specific and actionable points
example_feedback_improved = """
Feedback: The answer focuses primarily on QA applications but lacks discussion of other important tasks where retrieval-augmented LMs leverage both parametric and non-parametric knowledge. Question: How do language models combine parametric and non-parametric knowledge for text generation and summarization tasks?
Feedback: Only computational efficiency is mentioned as a limitation. The answer should include other significant drawbacks such as retrieval quality dependence and potential noise introduction. Question: What are the main limitations of combining parametric and non-parametric knowledge in language models?
Feedback: The response would benefit from better organization with clear subsections for advantages, applications, and limitations to improve readability.
Feedback: Add more quantitative results beyond the single Atlas example to strengthen the empirical evidence about the effectiveness of combining parametric and non-parametric approaches.
"""

# 优化后的示例答案，更好地展示期望的质量和结构
example_answer_rag_improved = """
Language models employ two distinct but complementary approaches to leverage knowledge: parametric and non-parametric methods, each with unique advantages and limitations for different applications.

**Parametric Knowledge Utilization**
Parametric knowledge refers to information encoded directly within a model's learned parameters during pre-training [0]. This approach enables models to perform closed-book question answering, where they generate responses based solely on internalized knowledge without accessing external sources during inference [0]. Large language models like GPT-3 demonstrate this capability effectively, achieving 64.3% accuracy on TriviaQA in zero-shot settings and improving to 71.2% in few-shot scenarios [3]. However, parametric knowledge has significant limitations, particularly for long-tail factual information. Research shows that even advanced models like GPT-3 davinci-003 fail to answer the majority of questions about less popular entities, with performance improvements from scaling plateauing for uncommon knowledge [4].

**Non-Parametric Knowledge Integration**
Non-parametric approaches address these limitations by retrieving relevant information from external document collections during inference [1]. Retrieval-augmented language models demonstrate substantial advantages, including reduced factual errors, improved attribution capabilities, and flexible knowledge updating without retraining [1]. The effectiveness of this approach is exemplified by Atlas, which achieves over 42% accuracy on Natural Questions using only 64 training examples, outperforming a 540B parameter model by 3% despite having 50x fewer parameters [2]. This demonstrates how retrieval augmentation can achieve superior performance with dramatically improved parameter efficiency.

**Challenges and Trade-offs**
Despite their benefits, retrieval-augmented approaches introduce computational overhead due to increased context length requirements [6]. Additionally, models may struggle to effectively utilize all retrieved information, particularly when relevant details are positioned in the middle of long contexts [6]. The choice between parametric and non-parametric approaches often depends on the specific application requirements, balancing factors such as inference efficiency, knowledge coverage, and the need for attributable responses.
"""

# Updated feedback prompt with improved structure
instruction_feedback_prompt = instruction_feedback.format_map(
    {
        "example_question": example_question_rag,
        "example_answer": example_answer_rag_incorrect,
        "example_feedback": example_feedback_improved,
    }
)

# Optimized feedback instance prompt with clearer instructions
feedback_example_instance_prompt = (
    instruction_feedback_prompt + "\nQuestion: {question}\n"
    "Answer: {answer}\n"
    "References (for context): {references}\n"
)


editing_feedback = """
We provide a question related to recent scientific literature, an answer from a strong language model, and feedback on the answer.

Your task is to incorporate the feedback to improve the answer while maintaining the original structure and content quality.

**Guidelines:**
1. **Selective Modification**: Only modify the parts that require enhancement as noted in the feedback. Keep other sentences unchanged.
2. **Content Preservation**: Do not omit any crucial information from the original answer unless the feedback explicitly specifies that certain sentences are incorrect and should be removed.
3. **Avoid Redundancy**: If you add new paragraphs or discussions, ensure you are not introducing repetitive content or duplicating ideas already included in the original response.
4. **Citation Integration**: Use existing references presented under "References" to support new discussions, referring to their citation numbers in the format [X].
5. **Structure Preservation**: Do not remove new lines or paragraphs in the original answer, unless the feedback specifically indicates that certain sentences are incorrect and should be removed, or the paragraph organization should be changed.
6. **Evidence-Based Enhancement**: When adding new information, base it on the provided references and cite appropriately.

**Output Format**: Your improved answer should be marked as [Response_Start] and [Response_End].

**Example:**

References:
[0] Title: Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models Text: Roberts et al. (2020) shows that T5 (Raffel et al., 2020) can perform a new task formulation, closedbook QA. Concretely, T5 can produce answers to questions without access to any corpus at inference time, instead producing answers based on its model parameters, tuned to remember information digested in pretraining.
[1] Title: Reliable, Adaptable, and Attributable Language Models with Retrieval Text: Unlike parametric LMs—which use large-scale text data only during training—retrieval-augmented LMs leverage an external large-scale collection of documents (datastore) at inference by selecting relevant documents from the datastore (Asai et al., 2023a). Retrieval-augmented LMs can W1: largely reduce factual errors (Mallen et al., 2023), W2: provide better attributions (Gao et al., 2023a), W3: enabling flexible opt-in and out of sequences (Min et al., 2024).
[2] Title: Atlas: Few-shot Learning with Retrieval Augmented Language Models Text: In this work we present Atlas, a carefully designed and pre-trained retrieval augmented language model able to learn knowledge intensive tasks with very few training examples. We perform evaluations on a wide range of tasks, including MMLU, KILT and NaturalQuestions, and study the impact of the content of the document index, showing that it can easily be updated. Notably, Atlas reaches over 42% accuracy on Natural Questions using only 64 examples, outperforming a 540B parameters model by 3% despite having 50x fewer parameters.
[3] Title: Language Models are Few-Shot Learners Text: Similarly, GPT-3 achieves 64.3% accuracy on TriviaQA in the zero-shot setting, 68.0% in the one-shot setting, and 71.2% in the few-shot setting, the last of which is state-of-the-art relative to fine-tuned models operating in the same closed-book setting.

Question: What are the advantages of retrieval-augmented LMs?

Original Answer: Retrieval-augmented LMs have been effective in various use cases, including reducing hallucinations [0] and are often more parameter-efficient than non retrieval-augmented LMs [2].

Feedback: The answer provides only a list of advantages without providing any concrete examples. Please provide more examples of how retrieval-augmented LMs have been used in practice.

Edited Answer:
[Response_Start]Retrieval-augmented LMs have been effective in various use cases, including reducing hallucinations [1] and are often more parameter-efficient than non retrieval-augmented LMs [2]. For instance, Atlas [2] achieves 42% accuracy on Natural Questions using only 64 examples, outperforming a 540B parameters model by 3% despite having 50x fewer parameters. Additionally, retrieval-augmented approaches demonstrate practical benefits such as largely reducing factual errors, providing better attributions, and enabling flexible opt-in and out of sequences [1].[Response_End]

---

Now, please apply these guidelines to improve the following answer based on the feedback provided:
"""

editing_instance_prompt = (
    editing_feedback
    + "\nReferences:\n{passages}\nQuestion: {question}\nOriginal Answer:\n{answer}\nFeedback:\n{feedback}\nEdited Answer:\n"
)


# 优化后的主要生成提示模板
prompts_w_references = (
    "You are an expert academic researcher. Provide a comprehensive, well-structured answer to the following research question based on the provided scientific literature. "
    "Follow these guidelines for an excellent response:\n\n"
    "**Content Requirements:**\n"
    "• Write multiple paragraphs (minimum 2-3) offering a thorough overview of the topic\n"
    "• Base your answer on evidence from multiple references rather than single sources\n"
    "• Synthesize information across papers to show relationships, similarities, and differences\n"
    "• Include specific details, methodologies, results, and implications where relevant\n"
    "• Organize content logically by themes, approaches, or chronological development\n\n"
    "**Citation Requirements:**\n"
    "• Cite all factual claims using the provided references in format [X]\n"
    "• Add citations at the end of relevant sentences: 'This approach shows effectiveness [1].'\n"
    "• For multiple supporting references, cite the most relevant one per statement\n"
    "• Only cite references that directly support your claims\n"
    "• Do not include author names, years, or full bibliographic details\n\n"
    "**Structure and Quality:**\n"
    "• Start with a clear topic introduction\n"
    "• Use smooth transitions between paragraphs\n"
    "• Conclude with implications or future directions if appropriate\n"
    "• Write in formal academic style suitable for researchers\n"
    "• Ensure clarity and coherence throughout\n\n"
    "**Output Format:** Mark your answer with [Response_Start] and [Response_End]\n\n"
    "**Example:**\n"
    "References:\n{example_passages}\n\n"
    "Question: {example_question}\n\n"
    "[Response_Start]{example_answer}[Response_End]\n\n"
    "Now answer the following question:\n"
)

prompts_w_references_v2 = """You are an expert academic researcher. Write a comprehensive, analytically rich, and academically rigorous response to the following research question, strictly based on the provided scientific literature. Your response will be evaluated along six critical dimensions: formal academic writing, clarity, non-redundancy, critical analysis, originality, and foresight.

Follow these instructions to ensure the highest standard of quality:

---

**[1] Substantive Content & Structure**

• Write at least **three logically structured paragraphs**, each focused on a distinct aspect of the topic (e.g., methodologies, findings, or debates).
• **Synthesize** evidence from multiple sources. Emphasize contrasts, agreements, or trends across the literature.
• **Critique** methodologies, assumptions, or conclusions where relevant. Offer alternatives or highlight key limitations.
• **Include technical details**: models, experiments, frameworks, or results when possible. Avoid vague generalizations.
• **Begin** with a clear topic introduction.
• **Ensure** smooth transitions between paragraphs.
• **Conclude** with implications or propose future directions where relevant.
• **Ensure** clarity and coherence throughout the text.

---

**[2] Language, Style, and Citations**

• Write in **formal academic style**: avoid conversational expressions, rhetorical questions, or ambiguous phrasing.
• Use **precise terminology** consistently and avoid redundancy. Each sentence should contribute unique value.
• Ensure **clarity and logical progression**. Use transitional phrases purposefully.
• Cite all factual claims using the format [X], where X is the reference number provided. Only cite when the claim is directly supported.
• Do **not include** author names, years, or bibliographic details.
• Avoid speculative claims unless clearly grounded in cited evidence.

---

**[3] Analytical & Forward-Looking Thinking**

• Provide **critical analysis** of the reviewed works: identify gaps, limitations, contradictions, or flawed assumptions.
• Offer **original insights**: propose new perspectives, unifying frameworks, or reinterpretations based on the literature.
• Identify **concrete, innovative directions for future research**, grounded in observed gaps or emerging questions. Proposals must be **specific and actionable**, not generic.

---

**Output must be suitable for publication in a peer-reviewed academic journal and will be evaluated on the following six dimensions:**
- Academic Formality (0–100)
- Clarity & Readability (0–100)
- Redundancy (0–100)
- Critical Analysis (0–100)
- Original Insights (0–100)
- Future Directions (0–100)

---

**Output Format:** Mark your answer with [Response_Start] and [Response_End]

**Example:**

References:
{example_passages}

Question:
{example_question}

[Response_Start]{example_answer}[Response_End]

Now answer the following question:
"""


prompts_w_references_v3 = """You are an expert academic researcher. Write a comprehensive, analytically rich, and academically rigorous response to the following research question, based strictly on the provided scientific literature. Your response will be evaluated on six dimensions: **academic formality, clarity, redundancy, critical analysis, originality, and future research insight.**

Follow the instructions below meticulously:

---

[1] **Content Organization & Structural Logic**

• Write **at least three logically cohesive paragraphs**, each addressing a distinct, non-overlapping thematic aspect (e.g., methods, findings, controversies).
• Within each paragraph:
  - Maintain **thematic focus**.
  - Ensure **literature synthesis**, not summary. Compare results, contrast perspectives, and identify patterns.
  - Provide **smooth transitions** between ideas and paragraphs. Avoid abrupt thematic jumps.
• Begin with a concise introduction framing the topic and scope. End with a **forward-looking conclusion**, proposing implications or future directions.

---

[2] **Language Precision, Style, and Conciseness**

• Maintain a **formal academic tone throughout**. Use technical terminology precisely and consistently. Avoid colloquial or speculative language.
• Each sentence must contribute **unique analytical value**. Repetition (even partial) without structural justification will result in penalties.
• Use **grammatically sophisticated sentence structures** to reflect complex reasoning. Vary sentence forms purposefully while maintaining clarity.

---

[3] **Evidence-Based Argumentation & Citation**

• Base every substantive claim on the provided literature, citing sources using format `[X]`, where X is the reference number.
• Avoid generalizations not supported by the references. Do **not include author names, years, or any bibliographic details**.
• Highlight **methodological strengths or flaws**, **contradictions**, or **debates** across sources. Point out **underexplored areas** where applicable.

---

[4] **Analytical Depth, Original Thinking, and Foresight**

• Provide **critical analysis**: point out weaknesses in reasoning, conflicting findings, or gaps in the literature. Avoid mere summary.
• Contribute **original insights**: propose novel syntheses, alternative interpretations, or new theoretical frameworks based on cited evidence.
• Suggest **specific, innovative research directions**, tightly linked to the literature gaps you have identified. Proposals should be **actionable and precise**, not general recommendations.

---

**Your output must be suitable for publication in a peer-reviewed academic journal and will be evaluated on six criteria:**
- Academic Formality (0–100)
- Clarity & Readability (0–100)
- Redundancy (0–100)
- Critical Analysis (0–100)
- Original Insights (0–100)
- Future Directions (0–100)

---

**Output Format:** Mark your response with [Response_Start] and [Response_End]

**Example:**

References:
{example_passages}

Question:
{example_question}

[Response_Start]{example_answer}[Response_End]

Now answer the following question:

"""


# 更新生成演示提示
generation_demonstration_prompts = prompts_w_references_v3.format_map(
    {
        "example_passages": example_passages_rag,
        "example_question": example_question_rag,
        "example_answer": example_answer_rag_improved,
    }
)

# 优化后的实例提示
generation_instance_prompts_w_references = (
    generation_demonstration_prompts + "References:\n{context}\n\n"
    "Question: {input}\n"
)

# 零样本版本优化
generation_instance_prompts_w_references_zero_shot = (
    "You are an expert academic researcher. Provide a comprehensive, well-structured answer to the research question based on the provided scientific literature.\n\n"
    "**Guidelines:**\n"
    "• Write multiple paragraphs offering thorough coverage of the topic\n"
    "• Synthesize information across multiple references to show relationships and differences\n"
    "• Include specific methodologies, results, and implications where relevant\n"
    "• Organize content logically by themes or approaches\n"
    "• Cite all factual claims using format [X] at the end of relevant sentences\n"
    "• Only cite references that directly support your statements\n"
    "• Write in formal academic style suitable for researchers\n"
    "• Mark your answer with [Response_Start] and [Response_End]\n\n"
    "References:\n{context}\n\n"
    "Question: {input}\n"
)

# 增强版提示，包含更详细的质量指标
generation_instance_prompts_w_references_enhanced = (
    "You are a senior academic researcher tasked with writing a comprehensive literature review response. "
    "Create an authoritative answer that demonstrates deep understanding of the research landscape.\n\n"
    "**Excellence Criteria:**\n"
    "1. **Comprehensiveness:** Cover all major aspects and approaches mentioned in the literature\n"
    "2. **Synthesis:** Connect findings across papers, highlighting convergent and divergent results\n"
    "3. **Critical Analysis:** Discuss strengths, limitations, and research gaps where appropriate\n"
    "4. **Methodological Insight:** Include details about experimental setups, datasets, and evaluation metrics\n"
    "5. **Future Implications:** Consider broader impact and future research directions\n\n"
    "**Structure Expectations:**\n"
    "• Opening: Clear problem definition and scope\n"
    "• Body: 2-4 well-organized paragraphs covering different aspects/approaches\n"
    "• Integration: Show how different studies relate to each other\n"
    "• Conclusion: Synthesize key insights and implications\n\n"
    "**Citation Standards:**\n"
    "• Every significant claim must be supported by appropriate citations [X]\n"
    "• Cite the most relevant reference for each statement\n"
    "• Maintain consistent citation placement at sentence endings\n"
    "• Only use provided references\n\n"
    f"{generation_demonstration_prompts}"
    "References:\n{context}\n\n"
    "Question: {input}\n"
)

judge_section_should_give_an_system = (
    "You are a helpful academic writing assistant that provides answers in JSON format."
)

judge_section_should_give_an_figure = """Determine if this academic paper section needs an image: "{section_name}"

Answer with a JSON object containing two fields:
1. "need_image": a string (yes/no)
2. "reason": a brief explanation (1-2 sentences)

Respond ONLY with valid JSON. Format:
{{"need_image": yes/no, "reason": "explanation"}}

Response:"""

format_reflection_prompt= """You are an expert scientific editor. Your task is to analyze the following text, which represents a section of a scientific paper, and ensure it adheres to standard academic writing formats and structure.

Analyze the text for:
- Coherence and logical flow between paragraphs.
- Appropriate paragraph breaks.
- Clear topic sentences for paragraphs.
- Overall structure suitable for a paper section.
- Avoidance of overly conversational or informal language.

**Constraint:** The text contains citations in the format `[number]`, like `[1]`, `[1][2][3]`. **It is absolutely critical that you DO NOT change the position or the numbers within these citations.** If you rewrite any part of the text, the citations must remain attached to the exact same preceding word or phrase as in the original text.

**Input Text:**

{section_content}

---

**Instructions:**
After analyzing the text, follow these rules strictly:

1. If you find significant formatting or structural issues that deviate from standard academic writing:
   - Rewrite the text to correct these issues.
   - **Strictly adhere to the constraint:** Preserve the exact position and content of all citations `[number]`.
   - Ensure the original meaning and information are fully preserved.
   - Output ONLY the rewritten text, enclosed within {RESPONSE_START_DELIMITER} and {RESPONSE_END_DELIMITER}.
2. If the text's format and structure are already appropriate for a scientific paper section:
   - Output the original text EXACTLY as provided, enclosed within {RESPONSE_START_DELIMITER} and {RESPONSE_END_DELIMITER}.

**Important:** Do not include any analysis, explanations, or commentary in your response. Output only the final text content between the delimiters below.

{RESPONSE_START_DELIMITER}
"""


context_refine_prompt = """
You will be given a markdown document. Please rewrite it according to the following requirements:

- Remove all headings (lines starting with #, ##, etc.).
- Remove any redundant or repetitive content.
- Remove any paragraph that appears to be a conclusion (e.g., starts with "In conclusion", "Therefore", "To summarize", etc.).
- Keep the main ideas and factual content intact.
- Rewrite the result in a clear, concise, and logically structured academic style suitable for a research paper section.
- Output the rewritten content as plain text only. Do not include explanations, markdown, or any commentary — only the revised version.

Original text:
\"\"\"
{markdown_text}
\"\"\"

Rewritten version (plain text only):
"""

validate_figure_caption_prompt = """
Evaluate whether the following figure caption is suitable for the section titled "{section_name}".
Caption content: {caption}

Please answer only "yes" or "no", and briefly state the reason (no more than one sentence).
Consider the following aspects for your evaluation:
- **Relevance:** Is the caption relevant to the section's theme ("{section_name}")?
- **Clarity & Conciseness:** Is the caption clear, concise, and easy to understand?
- **Completeness:** Does the caption adequately describe the figure's content?
- **Accuracy:** Is the information presented in the caption factually correct?
- **Grammar & Syntax:** Is the caption grammatically correct and well-formed?
- **Formatting Issues:** Does the caption contain unfinished LaTeX code, parsing error symbols, or other formatting problems?

- If the caption meets these criteria, answer "yes".
- If the caption fails on one or more significant criteria (e.g., irrelevant, unclear, contains major errors, incomplete, or has formatting issues), answer "no".
"""


template_extract_keywords_source_aware = """Extract optimal search keywords from the given research question, specifically optimized for '{source}' academic database. Your task is to generate concise, comma-separated query terms that will maximize relevant paper retrieval in this specific platform.

### Source-Specific Guidelines:

#### If targeting Semantic Scholar:
- Focus on technical terminology and core concepts
- Include methodological terms
- Consider author-centric keywords if prominent researchers are known
- Emphasize computer science and AI terminology where relevant

#### If targeting OpenAlex:
- Prioritize broader academic terms
- Include interdisciplinary connections
- Balance specificity with coverage
- Include field classifications where relevant

#### If targeting PubMed:
- Emphasize medical/biological terminology
- Include relevant MeSH (Medical Subject Headings) terms
- Consider clinical and biomedical contexts
- Include chemical/drug names or biological processes where relevant

### Format your response as follows:
[Start] keyword1, keyword2, keyword3, ...[End]

### Examples by Source:
- Semantic Scholar: [Start] transformer architecture, attention mechanism, language model fine-tuning[End]
- OpenAlex: [Start] neural networks, deep learning, artificial intelligence, pattern recognition[End]
- PubMed: [Start] CRISPR-Cas9, gene editing, genetic therapy, chromosomal modification[End]

Now, extract optimized search keywords for {source} from this question:
{user_query}
"""
