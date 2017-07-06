""" 
    Speeds up estimator.predict by preventing it from reloading the graph on each call to predict.
    It does this by creating a python generator to keep the predict call open.
    
    Usage: Just warp your estimator in a FastPredict. i.e.
    classifier = FastPredict(learn.Estimator(model_fn=model_params.model_fn, model_dir=model_params.model_dir))
    
    Author: Marc Stogaitis
 """

class FastPredict:
    
    def _createGenerator(self):
        while True:
            yield self.next_features

    def __init__(self, estimator):
        self.estimator = estimator
        self.first_run = True
        
    def predict(self, features):
        self.next_features = features
        if self.first_run:
            self.predictions = self.estimator.predict(x = self._createGenerator())
            self.first_run = False
        return next(self.predictions)
        
    