import numpy as np
from sklearn.base import ClassifierMixin
import pandas as pd


class SimpleAnomalyDetector(ClassifierMixin):
    """
    This class represent the simply anomaly detector used in our method.
    It is an autoencoder based model inspired by Antwarg (2019):
        @article{antwarg2021explaining,
        title={Explaining anomalies detected by autoencoders using SHAP},
        author={Antwarg, Liat and Miller, Ronnie Mindlin and Shapira, Bracha and Rokach, Lior},
        journal={Expert Systems with Applications},
        pages={115736},
        year={2021},
        publisher={Elsevier}
        }

    """

    def __init__(self, in_out_flag=True):
        self.truth_table = None
        self.truth_table_modified = None
        self.truthTableDict = None
        self.in_out_flag = in_out_flag
        self.input_idx = 0

    def fit(self, truth_table, truth_table_modified):
        """
        Receives the original and modified truth table and "trains" the model.

        Args:
            truth_table (dataFrame): original truth table of the circuit
            truth_table_modified (dataFrame): name of anomalous sys file to read
        """
        self.truth_table = truth_table # original
        self.truth_table_modified = truth_table_modified # anomalous
        self.input_idx = len([name for name in self.truth_table_modified.columns.values if name.startswith('i')]) # All inputs start with an 'i' 
        self.truthTableDict = {}
        for idx in range(self.truth_table_modified.shape[0]):
            true_record_input = str(self.truth_table_modified.iloc[idx, :self.input_idx].values)
            self.truthTableDict[true_record_input] = list(self.truth_table.iloc[idx, :].values)

    def predict(self, records, explain_output=None):
        """
        Receives a set of records or one record.
        Returns the output values from the original truth table.

        Args:
            records - the records to explain
            explain_output - the output to explain
        """
        records = records.astype(int)
        if len(records) > 1:
            predicted_array = []
            for i in range(len(records)):
                predicted = self.truthTableDict[str(records[i, :self.input_idx])]
                if explain_output is None:
                    predicted_array.append(predicted)
                else:
                    predicted_array.append(predicted[explain_output])
            return np.array(predicted_array)
        else:
            record_inputs = records[0, :self.input_idx]
            for idx in range(self.truth_table_modified.shape[0]):
                true_record_input = list(self.truth_table_modified.iloc[idx, :self.input_idx].values)
                if list(record_inputs) == true_record_input:
                    if explain_output is None:
                        return list(self.truth_table.to_numpy()[idx, :])
                    else:
                        return [int(self.truth_table.to_numpy()[idx, explain_output])]

    def predict_proba(self, records, explain_output=None):
        """
        Receives a set of records or one record and the index of the output that needs to be explained.
        Returns an array of probabilities for the explained output to get each possible value in the
        modified truth table.

        Args:
            records - the records to explain
            explain_output - the output to explain
        """
        if isinstance(records, pd.DataFrame):
            records = records.to_numpy()
        if len(records) > 1:
            predicted_array = []
            for i in range(len(records)):
                predicted = self.truthTableDict[str(records[i, :self.input_idx].astype(int))]
                if explain_output is None:
                    prob = [[0, 1] if predicted[j] == 1 else [1, 0] for j in range(len(predicted))]
                else:
                    pred = int(predicted[explain_output])
                    prob = [0, 1] if pred == 1 else [1, 0]
                predicted_array.append(prob)
            return np.array(predicted_array)
        else:
            record_inputs = records[0, :self.input_idx]
            for idx in range(self.truth_table_modified.shape[0]):
                true_record_input = list(self.truth_table_modified.iloc[idx, :self.input_idx].values)
                if list(record_inputs) == true_record_input:
                    if explain_output is None:
                        pred = list(self.truth_table.to_numpy()[idx, :])
                        prob = [[0, 1] if pred[j] == 1 else [1, 0] for j in range(len(pred))]
                    else:
                        pred = int(self.truth_table.to_numpy()[idx, explain_output])
                        prob = [0, 1] if pred == 1 else [1, 0]
                    return np.array(prob)
