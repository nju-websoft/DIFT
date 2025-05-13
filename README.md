# Are LLMs Really Knowledgeable for Knowledge Graph Completion?

> Knowledge Graph (KG) completion aims to infer new facts from existing knowledge. While recent efforts have explored leveraging large language models (LLMs) for this task, it remains unclear whether LLMs truly understand KG facts or how they utilize such knowledge in reasoning. In this work, we investigate these questions by proposing ProbeKGC, a benchmark dataset that reformulates KG completion as multiple-choice question answering with systematically controlled option difficulties. Empirical results show that LLMs often produce inconsistent answers when the same question is presented with varying distractor difficulty, suggesting a reliance on shallow reasoning such as elimination rather than genuine knowledge recall. To better quantify model confidence and knowledge grasp, we introduce Normalized Knowledge Divergence (NKD), a novel metric that complements accuracy by capturing distributional confidence in answer selection. We further analyze the influence of selection biases on LLM predictions and highlight that LLMs do not always fully exploit their stored knowledge. Finally, we evaluate three enhancement strategies and provide insights into potential directions for improving KG completion.

## Table of contents

1. [Dataset](#Dataset)
   1. [Overview](#overview)
   2. [Option Design](#Option)
   2. [Statistics](#Statistics)
   4. [Description](#Description)
2. [Evaluation and Analysis]()
   1. [Answer Consistency Evaluation](#Consistency)
   2. [Answer Confidence Quantification](#Confidence)
   3. [Knowledge Utilization Analysis](#Utilization)
   4. [Knowledge Enhancement Evaluation](#Enhancement)
3. [Code](#Code)
   1. [Description](#code_des)
   2. [Dependencies](#Dependencies)
   3. [Usage](#Usage)

## <h2 id='Dataset'>Dataset</h2>

The dataset can be downloaded from [here](https://drive.google.com/file/d/1fUfUsq7gQ3zhNZe1ZXRqD1PNOt9M3nWy/view?usp=drive_link). Extract it to the project root directory to use.

### <h3 id='overview'>Overview</h3>

We construct ProbeKGC based on three widely-used KG completion benchmark datasets: `FB15K237`, `WN18RR`, and `YAGO3-10`.
The statistics of these datasets are shown as follows:
| Datasets | #Entities | #Relations | #Training | #Validation | #Testing |
| :- | -: | -: | -: | -: | -: |
| FB15K237 | 14,541 | 237 | 272,115 | 17,535 | 20,466 |
| WN18RR | 40,943 | 11 | 86,835 | 3,034 | 3,134 |
| YAGO3-10 | 123,182 | 37 | 1,079,040 | 5,000 | 5,000 |

We randomly sample 2,000 test triplets from each dataset, producing three corresponding datasets: `ProbeKGC-FB`, `ProbeKGC-WN`, and `ProbeKGC-YG`. 
For each test triplet, we generate two KG completion queries: one for head entity prediction and one for tail entity prediction.
For each query, we design three sets of options corresponding to different option difficulty levels, and shuffle the options to avoid ranking bias.
We then convert each query and its four options into a natural language multi-choice question.
We ensure that each question contains exactly one correct answer among the four options, making it unambiguous and answerable.  

### <h3 id='Option'>Option Design</h3>

ProbeKGC includes three levels of option difficulty, each based on a different strategy for constructing distractors:
- **Easy**: Distractors are randomly sampled from the entire set of entities in the KG. 
- **Medium**: Distractors are sampled from entities of the same relation and type as the answer, based on relation semantics (i.e., the target entity types).
- **Hard**: Distractors are top-ranked entities semantically similar to the correct answer, selected using TransE embeddings.



### <h3 id='Statistics'>Statistics</h3>

The statistics of ProbeKGC are shown below.  

| Dataset        | Difficulty | #Entities | #Relations | #Triples | #Options | #Answer Type |
|----------------|------------|-----------:|------------:|----------:|----------:|---------------|
|                | Easy       | 9,427      | 161         | 2,000     | 4         | A/B/C/D       |
| **ProbeKGC-FB**    | Medium     | 7,753      | 161         | 2,000     | 4         | A/B/C/D       |
|                | Hard       | 5,940      | 161         | 2,000     | 4         | A/B/C/D       |
|                | Easy       | 13,028     | 11          | 2,000     | 4         | A/B/C/D       |
| **ProbeKGC-WN**    | Medium     | 12,257     | 11          | 2,000     | 4         | A/B/C/D       |
|                | Hard       | 10,981     | 11          | 2,000     | 4         | A/B/C/D       |
|                | Easy       | 14,686     | 30          | 2,000     | 4         | A/B/C/D       |
| **ProbeKGC-YG**    | Medium     | 13,831     | 30          | 2,000     | 4         | A/B/C/D       |
|                | Hard       | 11,109     | 30          | 2,000     | 4         | A/B/C/D       |


### <h3 id='Description'>Description</h3>

We take `ProbeKGC-FB` as an example to introduce the files in each dataset. The directory structure of each dataset is listed as follows:

```
ProbeKGC-FB/
├── entity.json: names and descriptions of entities
├── relation.json: statement templates for relations
├── train.txt: training triples
├── valid.txt: validation triples
├── test.txt: testing triples
├── test_simple.json: multi-choice questions with easy-level distractors
├── test_medium.json: multi-choice questions with medium-level distractors
├── test_hard.json: multi-choice questions with herd-level distractors
```


## <h2 id='Evaluation'>Evaluation and Analysis</h2>

### <h3 id='Consistency'>Answer Consistency Evaluation</h3>
Fig 1 presents the overall performance of different LLMs on ProbeKGC.

**Option Difficulty Influence**.
Option difficulty has a significant influence on model performance: accuracy drops sharply as the distractors become more challenging.
However, for ProbeKGC-WN, all models perform similarly on the *Easy* and *Medium* questions.  
This is because WN18RR is a lexical KG and there are only 11 relations describing basic lexical relationships between words.
As a result, even the distractors in the *Easy* setting often satisfy relation semantics, making the gap between *Easy* and *Medium* relatively small.

**Model Size Influence**. The model size contributes to LLM performance, by comparing LLaMA3.1-8B with LLaMA3.1-70B.
The performance gap is more obvious on the *Hard* questions.
A similar observation can be found with Qwen2.5-7B and Qwen2.5-72B.
The results indicate the superiority of larger models in answering difficult questions.

<div align="center">Fig.1 Accuracy for different LLMs on ProbeKGC</div>

![](figs/consistency.jpg)


**Answer Overlap Analysis**. Overall, we find that LLMs may provide different answers to the same question when presented with different option difficulty levels.
This suggests that LLMs lack robustness and tend to derive answers by eliminating incorrect options rather than confirm the answer from their internal knowledge of the target fact behind the question.
We further verify this hypothesis by obtaining the overlap of correctly answered questions with different option difficulties using Venn diagrams.
We take the results of GPT-4o mini as an example. As shown in Fig 2, the correctly answered questions exhibit a hierarchical inclusion relationship across the three levels of option difficulty. 
Specifically, in most cases, if a question with the *Medium* level of distractors is answered correctly, the corresponding *Easy* question is also answered correctly. 
A similar pattern holds for the relationship between the *Hard* and *Medium* settings.
Some questions can be only answered correctly with the *Easy* level of distractors, because the knowledge grasp of LLMs for the corresponding target facts is poor.
In contrast, some questions can be answered correctly with any option difficulty levels of distractors, indicating that LLMs have a better grasp of the related knowledge.

<div align="center">Fig.2 The overlap of correctly answered questions using GPT-4o mini</div>

![](figs/venn_gpt4o.jpg)


The overlap of correctly answered questions of GPT-3.5 Turbo, LLaMA3.1-8B, LLaMA3.1-70B, Qwen2.5-7B, and Qwen2.5-72B are shown is Fig.3, Fig.4, Fig.5, Fig.6, and Fig.7, respectively. We can find that the conclusions still hold here.

<div align="center">Fig.3 The overlap of correctly answered questions using GPT-3.5 Turbo</div>

![](figs/venn_gpt3.jpg)


<div align="center">Fig.4 The overlap of correctly answered questions using LLaMA3.1-8B</div>

![](figs/venn_llama3_8b.jpg)

<div align="center">Fig.5 The overlap of correctly answered questions using LLaMA3.1-70B</div>

![](figs/venn_llama3_70b.jpg)

<div align="center">Fig.6 The overlap of correctly answered questions using Qwen2.5-7B</div>

![](figs/venn_qwen_7b.jpg)

<div align="center">Fig.7 The overlap of correctly answered questions using Qwen2.5-72B</div>

![](figs/venn_qwen_72b.jpg)


**Findings**. This experiment suggests that LLMs may not truly grasp the knowledge of the questions, through the inconsistent answers of LLMs across different option difficulties.
LLMs heavily rely on shallow reasoning like the elimination strategy. Hence, accuracy is not comprehensive enough to evaluate whether LLMs grasp specific knowledge.


### <h3 id='Confidence'>Answer Confidence Quantification</h3>
As we have concluded, accuracy cannot adequately reflect whether LLMs grasp the knowledge.
Hereby, we seek new ways to assess the extent to which LLMs grasp the knowledge involved in the reasoning.
To this end, we start with the probability distribution of options. 
We hypothesize that an LLM's grasp of specific knowledge is associated with its confidence among options.
We argue that if an LLM has no knowledge about a particular question, the probability distribution across the options would be an inherent prior distribution. 
Otherwise, the LLM possesses sufficient knowledge to answer the question, the probability of the gold answer would be significantly higher than the other options. 
The greater the divergence of the LLM's probability distribution from the prior distribution, the more confident the LLM is in its grasp of the knowledge involved in reasoning.

We design a new metric based on the Kullback-Leibler (KL) divergence to measure the answer confidence and knowledge grasp of LLMs, called Normalized Knowledge Divergence (NKD). As shown in Fig.8, we compare the NKD results of LLMs under different option difficulties using radar charts and have the following observations.

**Accuracy vs. NKD**. By comparing LLaMA3.1-70B and Qwen2.5-72B, we find that although the accuracy scores of the two LLMs are similar, their NKD scores show a significant difference. A similar observation can be found when comparing LLaMA3.1-8B and Qwen2.5-7B. These findings indicate that NKD is different from accuracy.  Even if an LLM answers a question correctly, it may not be confident in its grasp of the knowledge. Besides, NKD is model-family and model-size dependent. GPT models tend to have the best confidence, followed by Qwen models, while LLaMA models show worst. For LLMs in the same family, larger LLMs are usually more confident in their grasp the knowledge.

<div align="center">Fig.8 NKD of different LLMs with different option difficulty levels</div>

![](figs/radar.jpg)

**Factors Affecting Accuracy and NKD**. We investigate the relationship between accuracy, NKD, and the degree of topic entities in questions (i.e., the reserved entities in KG queries), to understand the underlying factors affecting LLM performance. The entity degree is defined as the number of triplets in the KG that contains the entity. Entities with a high degree are referred to as popular entities, while those with a low degree are known as long-tail entities. Typically, popular entities appear more frequently in the pre-training corpora of LLMs, hence, LLMs may retain more knowledge related to them. We categorize KG completion questions based on the degree of the topic entities into different groups and analyze the accuracy and NKD scores for each group. The results on the *Hard* questions from ProbeKGC-FB are illustrated in Fig.9.

We focus on the *Hard* questions because they are less influenced by the shallow reasoning, and thus LLMs are more inclined to reason with the knowledge related to target facts. As the degree of entities increases, the accuracy of all LLMs shows a downward trend. It indicates that although LLMs retain more related knowledge for popular entities, the complexity of the knowledge makes it more difficult to retrieve and distinguish the related knowledge of target facts to answer the questions correctly. In contrast, all LLMs perform similarly and well for long-tail entities. We believe this is because, although there is less knowledge associated with long-tail entities, the knowledge is more direct and specific, facilitating easier understanding and grasp by the LLMs. Compared to accuracy, NKD scores show a certain improvement for popular entities. Even though LLMs have lower accuracy for questions about these entities, they are still confident due to the large amount of knowledge they have learned about these entities.

We present the results on ProbeKGC-WN and ProbeKGC-YG in Fig.10 and Fig.11, respectively. We can find that the accuracy of all LLMs shows a downward trend with the increasing of the entity degree on ProbeKGC-WN. Therefore, the conclusions obtained on ProbeKGC-FB still work. As for NKD, it exhibits a nearly monotonically decreasing trend as the entity degree increases. Differently, it does not show a slight improvement for popular entities. We think the reason is that ProbeKGC-WN is a lexical KG and the popular entities are often words with multiple meanings. LLMs are unable to exhibit strong confidence without additional knowledge which supports the reasoning. In contrast to ProbeKGC-FB, on ProbeKGC-YG, both accuracy and NKD show a increasing trend on popular entities, rather than monotonically decreasing. We think this is because that LLMs have a good grasp of popular entities on both ProbeKGC-FB and ProbeKGC-YG, however, the relations on ProbeKGC-YG are more straightforward, leading to a better reasoning performance.

<div align="center">Fig.9 Accuracy vs NKD on ProbeKGC-FB</div>

![](figs/line_FB.jpg)

<div align="center">Fig.10 Accuracy vs NKD on ProbeKGC-WN</div>

![](figs/line_WN.jpg)

<div align="center">Fig.11 Accuracy vs NKD on ProbeKGC-YG</div>

![](figs/line_YG.jpg)

**Findings**. The confidence of LLMs measured by our NKD metric shows significant differences, even though the accuracy of different LLMs is similar. This indicates that NKD can better reflect the different levels of knowledge grasp in LLMs compared to accuracy. LLM confidence is associated with their model-families and model sizes, and they are confident on both long-tail and popular entities. 


### <h3 id='Utilization'>Knowledge Utilization Analysis</h3>

We analyze how LLMs utilized knowledge for KG completion. It is acknowledged that LLMs are vulnerable to option position changes in multiple-choice questions due to their inherent selection bias. We seek to investigate whether LLMs leverage such biases or utilize their own knowledge for reasoning. Therefore, we move the gold answer of each question to every option position, creating four multiple-choice questions for LLMs to answer. We list the accuracy and NKD results on ProbeKGC-FB in Fig.12.

**Accuracy against Bias**. We observe that different LLMs exhibit a similar trend when the gold answer is sequentially placed in positions A through D, with accuracy generally declining from A to D (except for LLaMA3.1-8B). These results indicate that LLMs for KG completion tend to prefer the earlier options. This is an inherent bias, even though we do not provide explicit instructions to prompt the LLMs that the options are sorted by a KG completion model in the experimental setup, a practice that has been widely used in previous studies. This contrasts with the findings in the general domain, where the selection bias varies with different models and datasets.

**LLMs against Bias**. Moreover, there are significant differences in selection preferences within the same model family. For instance, LLaMA3.1-8B shows a strong preference for position B, while LLaMA3.1-70B prefers positions A and B. 
Similarly, Qwen2.5-7B distinctly favors position A, showing significant differences from other positions, while Qwen2.5-72B prefers both A and B.

**NKD against Bias**. As for our proposed NKD metric, although it is also affected by the selection bias, it proves to be more robust overall. The decrease in NKD scores from A to D is significantly less pronounced than that of accuracy. This suggests that NKD can somewhat resist the biases introduced by option position changes. It is a more reliable measure of model performance.

<div align="center">Fig.12 Performance w.r.t. option position bias on ProbeKGC-FB</div>

![](figs/bias_FB.jpg)

The results on ProbeKGC-WN and ProbeKGC-YG are shown in Fig.13 and Fig.14, respectively. Not surprisingly, the conclusions reached on ProbeKGC-FB still hold on these two datasets.

<div align="center">Fig.13 Performance w.r.t. option position bias on ProbeKGC-WN</div>

![](figs/bias_WN.jpg)

<div align="center">Fig.14 Performance w.r.t. option position bias on ProbeKGC-YG</div>

![](figs/bias_YG.jpg)

### <h3 id='Enhancement'>Knowledge Enhancement Evaluation</h3>

Given the differences of accuracy and NKD scores observed in previous experiments, we conclude that LLMs may indeed store knowledge related to KG facts, but they do not fully utilize it.  Therefore, we explore ways to improve their performance on KG completion from two aspects: improving the grasp of knowledge, and enhancing the utilization of internal knowledge.

**Enhancement Methods**. We introduce three strategies to enhance LLMs:
- **RAG**: We peroform retrieval-augmented generation (RAG) by using entity descriptions, entity-related triplets, and relation-related triplets together as the retrieved external knowledge.
- **CoT**: We prompt LLMs with chain-of-thought (CoT) to provide answers through step-by-step reasoning to improve knowledge utilization.
- **RAG & CoT**: We combine RAG with CoT together. We provide the LLMs with the retrieved knowledge and require them to perform step-by-step reasoning at the same time.

We further describe the RAG strategy used in our enhancement experiments. For each question, we retrieve 10 entity-related triplets and 10 relation-related triplets, and convert them into natural language statements. Entity-related triplets are sampled from the neighbor triplets of the topic entity, using the relation co-occurrence method proposed in DIFT. For relation-related triplets, we propose a heuristic ranking strategy: we maintain a candidate pool for each relation and prioritize triplets whose head and tail entities are less frequent (i.e., with lower total occurrence counts). In cases where multiple triplets have the same occurrence score, we rank them by the informativeness of their entities, measured by the sum of their node degrees. This process is repeated iteratively, re-sorting the remaining candidates after each selection.

**Result Analysis**. 
As shown in Fig.15, we find that CoT brings a significant performance improvement on ProbeKGC-FB, while RAG only provides a slight improvement or even a decrease in performance. However, the conclusion is the opposite on ProbeKGC-WN. For ProbeKGC-YG, in most cases, both RAG and CoT lead to improvements, with RAG providing a more significant boost. Due to that neither strategy consistently leads to stable improvements, the combination of RAG and CoT cannot reliably bring further performance gains. 


**Reason Analysis**. We attribute the performance differences of the strategies to the KG characteristics. From the perspective of knowledge categories, FB15K237 is an encyclopedic KG, and the internal knowledge of LLMs is sufficient to support reasoning. While both WN18RR and YAGO3-10 contain domain-specific knowledge (e.g., lexica and historical events), which LLMs may not be well-versed in. Therefore, the additional knowledge brought by RAG is highly beneficial. From the perspective of relations, the relations in FB15K237 are relatively complex, and LLMs can handle complex relationships with the help of CoT, which guides step-by-step thinking. In contrast, the relations in WN18RR and YAGO3-10 are more straightforward, and there is less dependence on step-by-step reasoning.  Additionally, the relations in YAGO3-10 are more diverse compared to WN18RR, which is why CoT also provides some improvement on YAGO3-10. As for the observations of the *Medium* level and the *Easy* level questions, they are similar to those of the **Hard** level on ProbeKGC-FB and ProbeKGC-YG. Therefore, the corresponding conclusions still hold. However, there are also some differences. We find that the improvement brought by RAG is comparable with that brought by CoT. This is because that the questions on ProbeKGC-WN in the *Medium* level and the *Easy* level are much easier that the *Hard* questions. The performance of accuracy is so high that the retrieved external knowledge cannot contribute too much, leading to a similar performance of RAG and CoT. We also notice that the combination of RAG and CoT sometimes leads to a decrease in performance compared to using no strategy at all. 

**Findings**. RAG and CoT offer complementary strengths in KG completion, depending on the characteristics of KGs. CoT is particularly effective for reasoning over complex relations, while RAG excels in supplementing external knowledge. For choosing a specific strategy to enhance LLMs, it is essential to consider the KG characteristics, such as knowledge categories and relation complexity.

<div align="center">Fig.15 Accuracy of different enhancement strategies</div>

![](figs/enhance.jpg)


## <h2 id='Code'>Code</h2>

### <h3 id='code_des'>Description</h3>

Folder "code" contains codes to reproduce our evaluation and analysis.
The directory structure is listed as follows:

```
ProbeKGC-FB/
├── accuracy.py: compute accuracy when using CoT
├── accuracy.py: compute accuracy and NKD score
├── chat_api.py: set your own API keys here
├── knowledge_graph.py: load the data of entities, relations, and the graph
├── main.py: answer the multi-choice questions via commercial APIs
├── main_open.py: answer the multi-choice questions via open-source LLMs when using CoT
├── main_open.py: answer the multi-choice questions via open-source LLMs
├── prompt_hub.py: define the prompt template
```

### <h3 id='Dependencies'>Dependencies</h3>

- Python=3.10.15
- ccelerate==1.1.1
- bitsandbytes==0.45.0
- fire==0.7.0
- matplotlib==3.10.0
- matplotlib-venn==1.1.1
- numpy==1.26.4
- openai==1.56.0
- peft==0.14.0
- scikit-learn==1.6.1
- torch==2.5.1
- transformers==4.45.2
- vllm==0.6.4

### <h3 id='Usage'>Usage</h3>
Definition of all hyper-parameters can be found in the corresponding code.

#### Obtain accuracy and NKD scores
For commercial APIs:
```
python main_api.py --dataset {ProbeKGC-FB} --data_mode {simple} --api_mode {gpt-4o-mini} --run_name {run_test}
```

For open-source LLMs:
```
python main_open.py --dataset {ProbeKGC-FB} --model_name {llama3.1-8b} --data_mode {simple} --run_name {run_test}
```

After calling LLMs, compute accuracy and NKD score:
```
python accuracy.py --dataset {ProbeKGC-FB} --data_model {simple} --run_name {run_test}
```

#### Bias Evaluation
For commercial APIs
```
python main_api.py --dataset {ProbeKGC-FB} --data_mode {simple} --api_mode {gpt-4o-mini} --run_name {run_test} --fixed_option_id {A}
```

answer the multi-choice questions with open-source LLMs
```
python main_open.py --dataset {ProbeKGC-FB} --model_name {llama3.1-8b} --data_mode {simple} --run_name {run_test} --fixed_option_id {A}
```

After calling LLMs, compute accuracy and NKD score:
```
python accuracy.py --dataset {ProbeKGC-FB} --data_model {simple} --run_name {run_test}
```

#### Knowledge Enhancement Evaluation
For commercial APIs
```
python main_api.py --dataset {ProbeKGC-FB} --data_mode {simple} --api_mode {gpt-4o-mini} --run_name {run_test} --logprobs {False}
```

answer the multi-choice questions with open-source LLMs
```
python main_open_cot.py --dataset {ProbeKGC-FB} --model_name {llama3.1-8b} --data_mode {simple} --run_name {run_test} 
```

After calling LLMs, compute accuracy and NKD score:
```
python accuracy_cot.py --dataset {ProbeKGC-FB} --data_model {simple} --run_name {run_test}
```