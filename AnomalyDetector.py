

class AnomalyDetector:
    def __init__(self, model):
        self.model = model

    def fit(self, *args):
        self.model.fit(*args)

    def predict(self, record, *args):
        return self.model.predict(record, *args)

    def predict_proba(self, record, *args):
        return self.model.predict_proba(record, *args)

