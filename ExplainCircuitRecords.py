import os
import time
import warnings

import numpy as np
import pandas as pd

from utils import *
from Explainers.KernelShap import KernelShap
from Explainers.SamplingShap import SamplingShap
from Explainers.Lime import Lime
import metrics
warnings.filterwarnings("ignore")


class ExplainCircuitRecords:
    """
    This class implements a method described in 'Evaluating anomaly explanations using the ground truth' to evaluate the correctness and robustness
    of explanations for anomalies produced by different explanation methods.

    Params:
    ----------
    anomaly_detector(AnomalyDetector): the anomaly detection model to use for identifying the anomalies

    in_out_flag (bool): indicate whether the input and output and concatenated to create 'auto encoder' format.
            If True the input and output are concatenated in each side, else one side contains the input
            and the other contains the output of the circuit

    amount_of_noise (int): the number of attribute noise to add to the data

    background_proportion (float): the proportion of data to assign to the background set
    """
    
    # Set the random seed used for randomly selecting a background set with the required proportion
    np.random.seed(27)

    def __init__(self, anomaly_detector, in_out_flag=True, amount_of_noise=0, background_proportion=1.0):
        self.AD_model = anomaly_detector
        self.in_out_flag = in_out_flag
        self.amount_of_noise = amount_of_noise
        self.background_proportion = background_proportion
        self.explainer = None
        self.truth_table = None
        self.truth_table_modified = None
        self.circuit_name = None
        self.labels = None
        self.output_idxs = None
        self.output_index = None
        self.input_names = None
        self.output_names = None
        self.input_idx = 0
        self.metrics = []

    def set_explainer(self, explainer):
        """
        Sets the explainer name to be used in the evaluation process.

        Args:
            explainer (str): the explainer name. Can be either KernelSHAP, LIME or SamplingSHAP, or implement a new method
         """
        self.explainer = explainer

    def set_circuit(self, circuit_name):
        """
        Sets the circuit name to be explained.

        Args:
            circuit_name (str): the circuit name. Can be either C17m 74182, 74283, 74181
        """
        self.circuit_name = circuit_name

    def set_metrics(self, _metrics):
        """
        Sets the metrics to calculate for the evaluation.

        Args:
            _metrics (list): the list of metrics. Can be either MRR, MAP, Accuracy or RPrecision
        """
        self.metrics = _metrics

    def read_data(self, circuit_name_with_anomalies):
        """
        Reads the anomalous truth table and separate the labels.

        Args:
            circuit_name_with_anomalies (str): the name of the modified circuit
        """
        data = pd.read_csv(
                os.path.join(TRUTH_TABLE_PATH + '/' + self.circuit_name + '/' + circuit_name_with_anomalies +
                             '_' + str(self.amount_of_noise) + 'noise' + '_in_labeled.csv')).astype(int)
        self.labels = data.loc[:, data.columns == 'label']
        self.truth_table_modified = data.loc[:, data.columns != 'label']
        return self.truth_table_modified

    def read_data_original(self):
        """
        Reads the original truth table.
        """
        self.truth_table = pd.read_csv(
            os.path.join(TRUTH_TABLE_PATH + '/' + self.circuit_name + '/' + self.circuit_name + '_' +
                         str(self.amount_of_noise) + 'noise' + '_out.csv')).astype(int)

    def fit(self, data):
        """
        "Train" the model- call the fit function of the AD model which receives circuit_name and circuit_name_modified
        and creates the dictionary that matches inputs and outputs of the "autoencoder"

        Args:
            data: data to be trained on (modified truth table)
        """
        self.read_data_original()
        self.AD_model.fit(self.truth_table, self.truth_table_modified)
        self.input_names = [name for name in self.truth_table_modified.columns.values if name.startswith('i')]
        self.output_names = [name for name in self.truth_table_modified.columns.values if name.startswith('o')]
        self.input_idx = len(self.input_names)
        self.output_idxs = [idx for idx, name in enumerate(self.truth_table_modified.columns) if name.startswith('o')]

    def get_top_anomaly_to_explain(self):
        """
        Find the anomalies by comparing the modified and original truth tables
        Returns:
            list: list of indexes of records in truth_table_modified to explain- this records
                  get different output values then in the original truth table (due to the change
                  in one of the inner circuits)
        """
        if self.truth_table is None:
            return None
        list_idx_to_explain = []
        for idx in range(self.truth_table_modified.shape[0]):
            row_output = list(self.truth_table_modified.iloc[idx, self.input_idx+self.amount_of_noise:].values)
            true_row_outputs = list(self.truth_table.iloc[idx, self.input_idx+self.amount_of_noise:].values)
            if row_output != true_row_outputs:
                list_idx_to_explain.append(idx)
        print('Number of anomalies found:', len(list_idx_to_explain))
        return list_idx_to_explain

    def get_features_with_highest_reconstruction_error(self, record_idx):
        """
        Find the outputs with different value then in original truth table

        Args:
            record_idx (int): The explained record index
        Returns:
            different_outputs_idx (list): List of int - index of outputs with different value then in original truth table
        """
        true_record_output = list(self.truth_table.iloc[record_idx, self.input_idx+self.amount_of_noise:].values)
        modified_record_output = list(self.truth_table_modified.iloc[record_idx, self.input_idx+self.amount_of_noise:].values)
        different_outputs_idx = [i + self.input_idx+self.amount_of_noise for i in range(len(true_record_output)) if
                                 true_record_output[i] != modified_record_output[i]]
        return different_outputs_idx

    def get_background_set(self, anomalies_idx=None):
        """
        Get random background_size records from truth table data and return it.
        The background_size is background_prec * 100.
        For example if the truth_table contain 100 records and background_prec is 0.1 then background_size is 10.
        Args:
            anomalies_idx (list) : Anomalies indexes
        Returns:
            background_set (data frame): Records from truth_table that will be the background set of the explanation
            of the record that we explain at that moment
        """
        if anomalies_idx is not None:
            rnd_idx = anomalies_idx
        else:
            background_size = int(self.truth_table_modified.shape[0] * min(self.background_proportion, 1))
            rnd_idx = np.random.randint(0, background_size, int(background_size * self.background_proportion))
        background_set = self.truth_table_modified.iloc[rnd_idx]
        for col in background_set.columns:
            background_set[col] = background_set[col].astype(int)
        return background_set

    def explain_unsupervised_data(self, circuit_name_modified):
        """
        First find top_records_to_explain by extracting all records in truth_table whose output is different in truth_table_modified
        The modification in the circuit affects the output of this records so we want to
        explain the differences and see if the explanation matches the modification.
        For each record in 'top_records_to_explain' we find 'features_to_explain_idx' list
        that contains all features (outputs) that got a different output in truth_table and
        truth_table_modified and for each one use the explanation method to explain its output.
        Each explanation is compared against the ground truth explanation to calculate the selected metrics.
        for example:
        record on truth_table:          i1: 1, i2: 1, i3: 0, o1: 0, o2: 0.
        record on truth_table_modified: i1: 1, i2: 1, i3: 0, o1: 0, o2: 1.
            * This record (i1: 1, i2: 1, i3: 0) will be in top_records_to_explain because
              the output in truth_table and truth_table_modified is different.
            * The feature (output) o2 will be in features_to_explain_idx because this feature
              is the only feature that got different value in truth_table and truth_table_modified

        Args:
            circuit_name_modified (string): name of the modified circuit
        Returns:
            method_final_scores (dict): the values of the selected correctness metrics
        """
        print('---Explaining anomalies---')
        records = []
        explanations_count = 0
        func_predict = lambda x: self.AD_model.predict(x, self.output_index)
        func_predict_proba = lambda x: self.AD_model.predict_proba(x, self.output_index)
        method_metrics = {'y_true': [], 'y_pred': []}
        method_values_df = pd.DataFrame()
        background_selection = 'Background- {}%'.format(self.background_proportion * 100)
        noise = '{} noise features'.format(self.amount_of_noise)

        print('---Read ground truth---')
        full_path_gt = os.path.join(TRACE_BACK_PATH + self.circuit_name + '/' + circuit_name_modified + '_ground_truth_'
                                    + str(self.amount_of_noise) + 'noise.csv')
        ground_truth_df = pd.read_csv(full_path_gt)
        all_records_gt = (ground_truth_df.set_index('Unnamed: 0'))['explanation for output']

        # --- can be replaced with a call to any anomaly detection method ---
        top_records_to_explain = self.get_top_anomaly_to_explain()
        if len(top_records_to_explain) == 0:
            return None

        background_set = self.get_background_set()
        anomalies_count = 0
        explain_time = 0
        self.create_folders(background_selection, noise)

        for record_idx in top_records_to_explain:
            record_to_explain = (self.truth_table_modified.loc[record_idx]).astype(int)
            records.append(record_to_explain.values)
            outputs_to_explain_idx = self.get_features_with_highest_reconstruction_error(record_idx)
            n_outputs_to_explain = len(outputs_to_explain_idx)
            method_values = [[] for _ in range(len(self.output_names))]
            features_names = [name for name in self.truth_table_modified.columns.values]

            for i in range(n_outputs_to_explain):
                self.output_index = outputs_to_explain_idx[i]
                output_ground_truth = eval(all_records_gt[record_idx])[record_to_explain.index[self.output_index]]

                start_time = time.time()
                if self.explainer == 'LIME':
                    explainer = Lime(background_set, features_names)
                    exp = explainer.explain(record_to_explain, func_predict_proba)
                    values = self.sort_lime(exp, features_names)

                elif self.explainer == 'KernelSHAP':
                    explainer = KernelShap(background_set, func_predict)
                    values = explainer.explain(record_to_explain)

                elif self.explainer == 'SamplingSHAP':
                    explainer = SamplingShap(background_set, func_predict)
                    values = explainer.explain(record_to_explain)

                else:
                    raise ValueError('unknown explanation method')
                    break
                # --- can add more explanation methods here ---

                explain_time += time.time() - start_time
                method_values[self.output_index - len(self.input_names) - self.amount_of_noise] = values

                y_true = np.array(self.id_encoder(output_ground_truth))
                y_pred = np.array(self.id_encoder(self.sorted_ids_by_method_values(values)))
                method_metrics['y_true'].append(y_true)
                method_metrics['y_pred'].append(y_pred)

                explanations_count += 1

            values_only_explained = [item for item in method_values if item != []]
            outputs_only_explained = ['o' + str(i + 1) for i, item in enumerate(method_values) if item!=[]]
            values_all_df = pd.DataFrame(data=values_only_explained, columns=self.truth_table.columns)
            values_all_df.insert(0, 'record index', [record_idx for _ in range(len(values_only_explained))])
            values_all_df.insert(1, 'explained output', outputs_only_explained)
            if method_values_df.empty:
                method_values_df = values_all_df
            else:
                method_values_df = method_values_df.append(values_all_df, ignore_index=True)

            anomalies_count += 1

        print(f'Explained {anomalies_count} anomalies')
        print(f'Explanation time: {explain_time/60} minutes')

        full_path_explained_method = os.path.join(
            METHOD_DICT[str(self.explainer)] + background_selection + '/' + noise + '/' +
            self.circuit_name + '/' + 'explain_' + circuit_name_modified + '.csv')
        method_values_df.to_csv(full_path_explained_method, index=False)

        method_final_scores = {}
        y_true = np.array(method_metrics['y_true'])
        y_pred = np.array(method_metrics['y_pred'])
        for metric in self.metrics:
            if metric == 'RPrecision':
                metric_val = metrics.calculate_MeanRPrecision(y_pred, y_true)
            elif metric == 'MAP':
                metric_val = metrics.calculate_MRR(y_pred, y_true)
            elif metric == 'MRR':
                metric_val = metrics.calculate_MAP(y_pred, y_true)
            elif metric == 'Accuracy':
                metric_val = metrics.calculate_accuracy(y_pred, y_true)
            else:
                raise ValueError('unknown metric')
            # --- can add more metrics here ---
            method_final_scores[metric] = [metric_val]
        return method_final_scores

    def create_folders(self, background_selection, noise):
        """
        Create the necessary folders in the file system
        """
        if self.explainer is None:
            raise ValueError('unknown explanation method')
        else:
            path = METHOD_DICT[self.explainer]
            outer_folder = os.path.join(path + background_selection)
            middle_folder = os.path.join(path + background_selection + '/' + noise)
            inner_folder = os.path.join(path + background_selection + '/' + noise + '/' + self.circuit_name)
            if not os.path.exists(path):
                os.makedirs(path)
            if not os.path.exists(outer_folder):
                os.makedirs(outer_folder)
            if not os.path.exists(middle_folder):
                os.makedirs(middle_folder)
            if not os.path.exists(inner_folder):
                os.makedirs(inner_folder)

    @staticmethod
    def get_circuit_tree(circuit_name, anomaly):
        """
        Reconstruct circuit tree from the tree file for a specific anomaly in the circuit
        Args:
            circuit_name (string): name of the circuit we explain
            anomaly (tuple of 3): anomaly details. anomaly[0] is the modified output,
            anomaly[1] is the the old gate, anomaly[2] is the new gate
        """
        circuit_tree_df = pd.read_csv(os.path.join(TRACE_BACK_PATH + circuit_name + '/' + circuit_name + '_tree.csv'))
        ct_dict = circuit_tree_df.to_dict()
        circuit_tree = {o: eval(op_input) for o, op_input in
                        zip(ct_dict['output'].values(), ct_dict['op_inputs'].values())}
        return circuit_tree

    @staticmethod
    def get_global_explanation(circuit_name):
        """
         Reconstruct the global explanation from the file

         Args:
             circuit_name - the name of the circuit
        """
        track_back = pd.read_csv(
            os.path.join(TRACE_BACK_PATH + circuit_name + '/' + circuit_name + '_global_explanation.csv'))
        tb_dict = track_back.to_dict()
        output_influence = {o: eval(contrib) for o, contrib in
                            zip(tb_dict['outputs'].values(), tb_dict['contributing inputs'].values())}
        return output_influence

    @staticmethod
    def id_encoder(values):
        """
        Encodes the inputs. For instance 'i1' is encoded to '1'

        Args:
            values - the inputs to encode
        Returns:
            a list of encoded values
       """
        return [int(val[1:]) for val in values]

    @staticmethod
    def sorted_ids_by_method_values(values):
        """
        Returns the features ids sorted descending according to the method values for all non-zero valued features

        Args:
            values (array): values of an explanation method for a specific record
        """
        sorted_values = [idx for idx in np.flipud(np.argsort(np.abs(values))) if values[idx] != 0]
        top_values = ['i' + str(i + 1) for i in sorted_values if i != 0]
        return top_values

    @staticmethod
    def sort_lime(exp, features_names):
        """
        Sorts the explanation produces by LIME in descending order

        Args:
            exp: the unsorted explanation
            features_names : the names of the features in the explanation
        """
        lime_features = []
        lime_features_strings = [row[0] for row in exp.as_list()]
        for feature_string in lime_features_strings:
            del_idx = feature_string.find('i')
            if del_idx < 0:
                del_idx = feature_string.find('o')
            if del_idx < 0:
                del_idx = feature_string.find('n')
            feature_name = feature_string[del_idx:del_idx + 3].replace(' ', '')
            lime_features.append(feature_name)
        lime_values = [row[1] for row in exp.as_list()]
        values = np.ndarray(len(features_names))
        for j, col in enumerate(features_names):
            col_idx = lime_features.index(col)
            values[j] = lime_values[col_idx]
        return values


