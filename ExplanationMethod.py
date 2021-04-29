

class ExplanationMethod:
    """
        This class is the basic class that every wrapper to an explanation method should derive from and
        implement the explain function.
    """
    def __init__(self, *args):
        self.explainer = None

    def explain(self, records, *args):
        self.explainer.explain(records, *args)

