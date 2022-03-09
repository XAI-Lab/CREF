from ExplanationMethod import ExplanationMethod
import shap


class SamplingShap(ExplanationMethod):
    """
    This class is a wrapper to the Sampling Explainer of SHAP

    @incollection{NIPS2017_7062,
        title = {A Unified Approach to Interpreting Model Predictions},
        author = {Lundberg, Scott M and Lee, Su-In},
        booktitle = {Advances in Neural Information Processing Systems 30},
        editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
        pages = {4765--4774},
        year = {2017},
        publisher = {Curran Associates, Inc.},
        url = {http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf}
        }

    Args:
        func_predict: the predict function of the anomaly detector
        background_set: background set sampled from the train dataset
    """

    def __init__(self, func_predict, background_set):
        self.explainer = shap.SamplingExplainer(func_predict, background_set)

    def explain(self, records_to_explain, nsamples=500):
        """
        Args:
            records_to_explain: The set of records or a single record to explain
            nsamples: "auto" or int. Number of times to re-evaluate the model when explaining each prediction. M
            ore samples lead to lower variance estimates of the SHAP values. The "auto" setting uses
            `nsamples = 2 * X.shape[1] + 2048`. Default value here is set to 500.
        Return:
            SHAP values for every record
        """
        values = self.explainer.shap_values(records_to_explain, nsamples=nsamples)
        return values

    def __str__(self):
        return 'SamplingSHAP'


