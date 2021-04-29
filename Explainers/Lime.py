from lime import lime_tabular
from ExplanationMethod import ExplanationMethod


class Lime(ExplanationMethod):
    """
    This class is a wrapper to the LIME Explainer

    @inproceedings{ribeiro2016should,
      title={" Why should i trust you?" Explaining the predictions of any classifier},
      author={Ribeiro, Marco Tulio and Singh, Sameer and Guestrin, Carlos},
      booktitle={Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining},
      pages={1135--1144},
      year={2016}
    }
    Args:
        background_set: background set sampled from the dataset
        features_names: feature names of the features in the dataset
    """

    def __init__(self, background_set, features_names):
        self.features_names = features_names
        self.explainer = lime_tabular.LimeTabularExplainer(background_set, mode='classification',
                                                           feature_names=self.features_names)

    def explain(self, records_to_explain, func_predict_proba):
        """
        Args:
            records_to_explain: The set of records or a single record to explain
            func_predict_proba: the predict_proba function of the anomaly detector
        Return:
            values for every record
         """
        features_len = len(self.features_names)
        values = self.explainer.explain_instance(records_to_explain, func_predict_proba, num_features=features_len)
        return values

    def __str__(self):
        return 'LIME'

