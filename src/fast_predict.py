""" 
    Speeds up estimator.predict by preventing it from reloading the graph on each call to predict.
    It does this by creating a python generator to keep the predict call open.
    
    Usage: Just warp your estimator in a FastPredict. i.e.
    classifier = FastPredict(learn.Estimator(model_fn=model_params.model_fn, model_dir=model_params.model_dir))
    
    Author: Marc Stogaitis
 """

class FastPredict:
    
    def _createGenerator(self):
        while not self.closed:
            yield self.next_features
    
    def __init__(self, estimator):
        self.estimator = estimator
        self.first_run = True
        self.closed = False
        
    def predict(self, features):
        self.next_features = features
        if self.first_run:
            self.batch_size = len(features)
            self.predictions = self.estimator.predict(x = self._createGenerator(), batch_size=None)
            self.first_run = False
        elif self.batch_size != len(features):
            raise ValueError("All batches must be of the same size. First-batch:" + str(self.batch_size) + " This-batch:" + str(len(features)))
        
        results = []
        for _ in range(self.batch_size):
            results.append(next(self.predictions))
        return results 
        
    
    def close(self):
        self.closed=True
        next(self.predictions)