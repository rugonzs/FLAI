

# FLAI : Fairness Learning in Artificial Intelligence

[![Upload Python Package](https://github.com/rugonzs/FLAI/actions/workflows/python-publish.yml/badge.svg)](https://github.com/rugonzs/FLAI/actions/workflows/python-publish.yml)

[![Upload Docs](https://github.com/rugonzs/FLAI/actions/workflows/docs.yml/badge.svg)](https://github.com/rugonzs/FLAI/actions/workflows/docs.yml)

[![PyPI Version](https://img.shields.io/pypi/v/flai-causal)](https://pypi.org/project/flai-causal/)

[![Downloads](https://static.pepy.tech/badge/flai-causal)](https://pepy.tech/project/flai-causal)

[![Downloads](https://static.pepy.tech/badge/flai-causal/month)](https://pepy.tech/project/flai-causal)

[![Downloads](https://static.pepy.tech/badge/flai-causal/week)](https://pepy.tech/project/flai-causal)

Python library developed by Rubén González during his phD. research. His mission? To mitigate bias and discrimination through the application of causal algorithms.

[Demo](https://www.rubengonzalez.ai/demo)

[Documentation](https://rugonzs.github.io/FLAI/)

## Overview

![Overview](https://github.com/rugonzs/FLAI/blob/main/Documents/fair_algorithm.svg)

**FLAI** is a Python library designed with two key functionalities: building a **causal algorithm** and **mitigating biases** within it.

1. **Causal Algorithm Creation:** This library facilitates the development of a reliable causal algorithm, setting the stage for impartial data analysis.

2. **Bias Mitigation:** Fairness is pursued in two significant areas - **In-Training** and **Pre-Training**.

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

## Features

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

##### Shap Results
```python
import shap

explainer_original = shap.Explainer(model_original)
explainer_mitigated = shap.Explainer(model_mitigated)
shap_values_orignal = explainer_original(original_dataset.data[
                                               ['sex', 'race','age','education']])
shap_values_mitigated = explainer_mitigated(original_dataset.data[
                                               ['sex', 'race', 'age','education']])
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
