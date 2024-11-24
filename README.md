

# FLAI : Fairness Learning in Artificial Intelligence

[![Upload Python Package](https://github.com/rugonzs/FLAI/actions/workflows/python-publish.yml/badge.svg)](https://github.com/rugonzs/FLAI/actions/workflows/python-publish.yml)

[![Upload Docs](https://github.com/rugonzs/FLAI/actions/workflows/docs.yml/badge.svg)](https://github.com/rugonzs/FLAI/actions/workflows/docs.yml)

[![PyPI Version](https://img.shields.io/pypi/v/flai-causal)](https://pypi.org/project/flai-causal/)

[![Downloads](https://static.pepy.tech/badge/flai-causal)](https://pepy.tech/project/flai-causal)

[![Downloads](https://static.pepy.tech/badge/flai-causal/month)](https://pepy.tech/project/flai-causal)

[![Downloads](https://static.pepy.tech/badge/flai-causal/week)](https://pepy.tech/project/flai-causal)

Python library developed by Rubén González during his phD. research. His mission? To mitigate bias and discrimination through the application of causal algorithms. Use the citation at the bottom to reference this work.

[Demo](https://www.rubengonzalez.ai/demo)

[Documentation](https://rugonzs.github.io/FLAI/)

## Overview

![Overview](https://github.com/rugonzs/FLAI/blob/main/Documents/fair_algorithm.svg)

**FLAI** is a Python library designed with three key functionalities: detect **discrimination**, building a **causal algorithm**, and **mitigating biases** within it.

1. **Fairness Metric: Bias Detection**, This library introduces a novel metric to measure Fairness, which consists of a two-dimensional vector: Equality and Equity. This metric allows proper quantification of both justice dimensions.

2. **Causal Algorithm Creation:** This library facilitates the development of a reliable causal algorithm, setting the stage for impartial data analysis.

3. **Bias Mitigation:** Fairness is pursued in two significant areas - **In-Training** and **Pre-Training**.

    ### In-Training Mitigation

    The library includes features that allow the user to adjust the causal algorithm in two essential ways:

    - **Graph Relationship Modification:** Relationships within the graph can be modified to establish a more balanced structure.
    - **Probability Table Modification:** The probability table can be adjusted to prevent propagation or amplification of existing biases.

    ### Pre-Training Mitigation

    With the mitigated causal algorithm, a bias-free dataset can be generated. This dataset can be used for the training of other algorithms, enabling the bias mitigation process to extend to the initial stages of new model development.




## Installation

**FLAI** can be easily installed using pip, Python's package installer. Open your terminal or command prompt and type the following command:

```bash
pip install flai-causal
```

python <= 3.9
```bash
pip install flai-causal==2.0.0
```

## Features Fairnes Metric

### Single sensible feature

Measure equality and equity.

```python
from FLAI import data
from FLAI import causal_graph
import pandas as pd

df = pd.read_parquet('../Data/case_1.parquet')
flai_dataset = data.Data(df, transform=False)

df_f,datos_f = flai_dataset.fairness_eqa_eqi(features = ['education'], 
                              target_column = 'proba', 
                              column_filter = ['sensible'],
                              plot = True)
```

Group is the underprivileged group, in this case sensible = 0. The reference is the privileged group, in this case sensible = 1

|   Group            |   reference        | Equity  | Equality |   Fairness   |
|--------------------|--------------------|---------|----------|--------------|
| sensible 0         |    sensible 1      | -004    | 0.08     |   0.09       |


![Original Graph](https://github.com/rugonzs/FLAI/blob/main/Documents/fairness_metric.svg)


### Multiple Sensible Feature


```python
df_f,datos_f = flai_dataset.fairness_eqa_eqi(features = ['education','age'], 
                              target_column = 'proba', 
                              column_filter = ['race','sex'],
                              plot = True)
```

|   Group            |   reference        | Equity  | Equality |   Fairness   |
|--------------------|--------------------|---------|----------|--------------|
| race_0 sex_0       |    race_1 sex_1    | -0.06   |   0.18   |    0.19      |
| race_0 sex_1       |    race_1 sex_1    | -0.04   |   0.06   |    0.07      |
| race_1 sex_0       |    race_1 sex_1    | -0.07   |   0.15   |    0.17      |


![Original Graph](https://github.com/rugonzs/FLAI/blob/main/Documents/fairness_metric_orig.svg)


## Features Mitigation

### Causal Creation

```python
from FLAI import data
from FLAI import causal_graph
import pandas as pd

df = pd.read_pickle('../Data/adult.pickle')
flai_dataset = data.Data(df, transform=True)
flai_graph = causal_graph.CausalGraph(flai_dataset, target = 'label')
flai_graph.plot(directed = True)
```
![Original Graph](https://github.com/rugonzs/FLAI/blob/main/Documents/original_graph.svg)


### Causal Mitigation

#### Relations Mitigation

```python

flai_graph.mitigate_edge_relation(sensible_feature=['sex','age'])
```
![Mitigated Graph.](https://github.com/rugonzs/FLAI/blob/main/Documents/mitigated_graph.svg)

#### Table Probabilities Mitigation

```python

flai_graph.mitigate_calculation_cpd(sensible_feature = ['age','sex'])

```

#### Inference

Assess the impact of sensitive features before mitigation. Sex, Age and Label 0 is the unfavorable value.

```python
flai_graph.inference(variables=['sex','label'], evidence={})
flai_graph.inference(variables=['age','label'], evidence={})

```

|   sex | label |   p    |
|-------|-------|--------|
|   0   |   0   | 0.1047 |
|   **0**   |   **1**   | **0.2053** |
|   1   |   0   | 0.1925 |
|   **1**   |   **1**   | **0.4975** |

| age | label |   p    |
|-----|-------|--------|
|  0  |   0   | 0.0641 |
|  **0**  |   **1**   | **0.1259** |
|  1  |   0   | 0.2331 |
|  **1**  |   **1**   | **0.5769** |

Assess the impact of sensitive features after mitigation. Changes in sex or age not affect the output.

```python
mitigated_graph.inference(variables=['sex','label'], evidence={})
mitigated_graph.inference(variables=['age','label'], evidence={})

```

| sex | label |   p    |
|-----|-------|--------|
|  0  |   0   | 0.1498 |
|  **0** |   **1**   | **0.3502** |
|  1  |   0   | 0.1498 |
|  **1** |   **1**   | **0.3502** |


| age | label |   p    |
|-----|-------|--------|
|  0  |   0   | 0.1498 |
|  **0**  |   **1**   | **0.3502** |
|  1  |   0   | 0.1498 |
|  **1**  |   **1**   | **0.3502** |


### Fair Data

```python
fair_data = flai_graph.generate_dataset(n_samples = 1000, methodtype = 'bayes')
```
![Correlation in original and Fair Data.](https://github.com/rugonzs/FLAI/blob/main/Documents/corr.svg)

#### Train Algorithm With Fair Data.

```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

mitigated_X = fair_data.data[['age', 'sex', 'credit_history','savings','employment' ]]
mitigated_y = fair_data.data[['label']]
mitigated_X_train, mitigated_X_test, mitigated_y_train, mitigated_y_test = train_test_split(mitigated_X,
                                                           mitigated_y, test_size=0.7, random_state=54)
model_mitigated = XGBClassifier()
model_mitigated.fit(mitigated_X_train, mitigated_y_train)
metrics = mitigated_dataset.fairness_metrics(target_column='label', predicted_column = 'Predicted',
                            columns_fair = {'sex' : {'privileged' : 1, 'unprivileged' : 0},
                                            'age' : {'privileged' : 1, 'unprivileged' : 0}})
```

##### Metrics Performance

|                 |   ACC  |   TPR   |   FPR   |   FNR   |   PPP   |
|-----------------|--------|---------|---------|---------|---------|
| model           | 0.7034 | 0.97995 | 0.94494 | 0.02005 | 0.96948 |
| sex_privileged  | 0.7024 | 0.97902 | 0.94363 | 0.02098 | 0.96841 |
| sex_unprivileged| 0.7044 | 0.98087 | 0.94626 | 0.01913 | 0.97055 |
| age_privileged  | 0.7042 | 0.97881 | 0.94118 | 0.02119 | 0.96758 |
| age_unprivileged| 0.7026 | 0.98109 | 0.94872 | 0.01891 | 0.97139 |

##### Metrics Fairness

|                 |   EOD   |   DI    |   SPD   |   OD    |
|-----------------|---------|---------|---------|---------|
| sex_fair_metrics| 0.00185 | 1.00221 | 0.00214 | 0.00448 |
| age_fair_metrics| 0.00228 | 1.00394 | 0.00382 | 0.00981 |

#### Equity, Equality and Fairness Curve

```python
df_f,datos_f = fair_data.fairness_eqa_eqi(features = ['education','age'], 
                              target_column = 'proba', 
                              column_filter = ['race','sex'],
                              plot = True)
```

|   Group            |   reference        | Equity  | Equality |   Fairness   |
|--------------------|--------------------|---------|----------|--------------|
| race_0 sex_0       |    race_1 sex_1    | 0.0     |   0.0    |    0.0       |
| race_0 sex_1       |    race_1 sex_1    | 0.0     |   0.0    |    0.0       |
| race_1 sex_0       |    race_1 sex_1    | 0.0     |   0.0    |    0.0       |

![Original Graph](https://github.com/rugonzs/FLAI/blob/main/Documents/fairness_metric_mit.svg)

##### Shap Results
```python
import shap

explainer_original = shap.Explainer(model_original)
explainer_mitigated = shap.Explainer(model_mitigated)
shap_values_orignal = explainer_original(original_dataset.data[
['age', 'sex', 'credit_history','savings','employment']])
shap_values_mitigated = explainer_mitigated(original_dataset.data[
 ['age', 'sex', 'credit_history','savings','employment']])
shap.plots.beeswarm(shap_values_orignal)
shap.plots.bar(shap_values_orignal)    

shap.plots.beeswarm(shap_values_mitigated)
shap.plots.bar(shap_values_mitigated)

```
![shap values.](https://github.com/rugonzs/FLAI/blob/main/Documents/shap_o.svg)

![shap values.](https://github.com/rugonzs/FLAI/blob/main/Documents/shap_m.svg)


### References
* https://erdogant.github.io/bnlearn/
* http://pgmpy.org
## Citation
[Fairness Metric Paper Soon....]()

[Mitigation Paper](https://www.sciencedirect.com/science/article/pii/S0167739X24000694)
```
@article{GONZALEZSENDINO2024384,
title = {Mitigating bias in artificial intelligence: Fair data generation via causal models for transparent and explainable decision-making},
journal = {Future Generation Computer Systems},
volume = {155},
pages = {384-401},
year = {2024},
issn = {0167-739X},
doi = {https://doi.org/10.1016/j.future.2024.02.023},
url = {https://www.sciencedirect.com/science/article/pii/S0167739X24000694},
author = {Rubén González-Sendino and Emilio Serrano and Javier Bajo},
keywords = {Causal model, Bias mitigation, Fairness, Responsible artificial intelligence, Bayes},
abstract = {In the evolving field of Artificial Intelligence, concerns have arisen about the opacity of certain models and their potential biases. This study aims to improve fairness and explainability in AI decision making. Existing bias mitigation strategies are classified as pre-training, training, and post-training approaches. This paper proposes a novel technique to create a mitigated bias dataset. This is achieved using a mitigated causal model that adjusts cause-and-effect relationships and probabilities within a Bayesian network. Contributions of this work include (1) the introduction of a novel mitigation training algorithm for causal model; (2) a pioneering pretraining methodology for producing a fair dataset for Artificial Intelligence model training; (3) the diligent maintenance of sensitive features in the dataset, ensuring that these vital attributes are not overlooked during analysis and model training; (4) the enhancement of explainability and transparency around biases; and finally (5) the development of an interactive demonstration that vividly displays experimental results and provides the code for facilitating replication of the work.}
}
```