# Evaluating anomaly explanations using ground truth

**This repository includes python source code for CREF: **C**orrectness and **R**obustness **E**valuation **F**ramework based on ground truth explanations.**

CREF includes:
1. A real-world dataset with anomalies, based on ISCAS '85 and 74x series benchmarks [[1]](#1).
2. Ground truth global and local explanations.
3. Source code for the correctness and robustness evaluation framework.

_____________________________________________________________________________________________________________________________________

*The dataset can be downloaded using the following link: [Download](https://doi.org/10.7910/DVN/W4FPPN)*


:oil_drum: Dataset
-------
The dataset "Digital circuits dataset for anomaly detection + ground truth explanations" was created based on 4 digital circuits from ISCAS '85 and 74x series benchmarks [[1]](#1):
- C17
- 74182
- 74183
- 74181
The dataset includes 2 folders:
1. **Truth_Tables**: For each circuit we created 4 model based anomalous versions by changing a single gate to its negation. 
In addition, each anomalous version was padded with a varying amount of attribute noise (2, 4 and 6 noise features). 
The folder includes a truth table for both the original and the anomalous versions, with noise and without noise. The truth tables serve as binary tabular data.
Overall, there are 64 anomalous truth tables with labels and 16 original truth tables.
For example:
  * ```c17_modified_z4_from_nand2_to_and2_0noise_in_labeled.csv``` - a labeled truth table with anomalies of circuit c17 after negating gate z4, with no noise added.
  * ```c17_0noise_out.csv```- the original truth table of circuit c17

2. **Ground_Truth**: Each truth table has a respective file of local ground truth explanations. The folder also includes a global ground truth explanation for each circuit. 
For example:
  * ```c17_modified_z4_nand2_to_and2_ground_truth_0noise.csv``` - a file with the local ground truth explanations for every anomaly in the respective truth table.
  * ```c17_global_explanation.csv``` - a file with the global ground truth explanation for circuit c17.

The Truth_Tabels and Ground_Truth files are divided into subfolders of each circuit. 


:detective: Ground truth explanations
-------
A **local** ground truth explanation is the reason why a model returned a certain prediction for a specific instance. 
Such an explanation may be represented as the set of features that led the model to make that prediction. 
A **global** ground truth explanation provides an explanation for the entire system's behavior.
The ground truth explanations can enable the evaluating the correctness and robustness of explanations produced by an explanation method, as proposed by this framework.


:mag_right: Evaluate local explanations
------------
The proposed ground truth based framework aims to enable users to evaluate explanations produced by explanation methods using correctness and robustness metrics.
The framework, in its current settings, allow detecting anomalies with an autoencoder based anomaly detector, and then explain the anomalies using three common model-agnostic explanation methods:
  * ```Kernel SHAP``` [[2]](#2)
  * ```Sampling SHAP``` [[2]](#2)
  * ```LIME``` [[3]](#3)
The local explanations can be evaluated for correctness and robustness using several metrics (including MRR, MAP, R-precision, accuracy etc.), which are calculated by comparing the explanations to the ground truth explnations.

___________________________________________________________________________________________________________________________________________________________________________\

:receipt: Usage
------------------
1. Clone the repository.
2. Download the dataset, unzip it and place it in the top folder of cloned project.
3. Run the code

:pencil2: **Possible changes:**
- Implement you own anomaly detector instead of the method we used. Create a new class that derives from the class 'AnomalyDetector', implement the *constructor, fit, predict* and *predict_proba* methods.
- Implement an additional explanation method. Create a new class that derives from the class 'ExplanationMethod', implement the *constructor* and *explain* methods.
- Use different evluation metrics from the metrics offers in the 'metrics.py' file or add new metrics.

Follow the comments in the *ExplainAnomalies* main file for the adjustments that can be done to change any of the above components.

___________________________________________________________________________________________________________________________________________________________________________\

## References
<a id="1">[1]</a> 
Bryan, David. (1985). 
The ISCAS '85 benchmark circuits and netlist format directory.
[original benchmark:](https://people.engr.ncsu.edu/brglez/CBL/benchmarks/ISCAS85)

<a id="2">[2]</a> 
Lundberg, Scott, and Su-In Lee. "A unified approach to interpreting model predictions." arXiv preprint arXiv:1705.07874 (2017).

<a id="3">[3]</a> 
Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin. "" Why should i trust you?" Explaining the predictions of any classifier." Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining. (2016).
