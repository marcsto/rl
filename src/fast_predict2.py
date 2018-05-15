"""
    Speeds up estimator.predict by preventing it from reloading the graph on each call to predict.
    It does this by creating a python generator to keep the predict call open.

    Usage: Just warp your estimator in a FastPredict. i.e.
    classifier = FastPredict(learn.Estimator(model_fn=model_params.model_fn, model_dir=model_params.model_dir), my_input_fn)

    This version supports tf 1.4 and above and can be used by pre-made Estimators like tf.estimator.DNNClassifier. 

    Author: Marc Stogaitis
 """
import tensorflow as tf


class FastPredict:

    def __init__(self, estimator, input_fn):
        self.estimator = estimator
        self.first_run = True
        self.closed = False
        self.input_fn = input_fn

    def _create_generator(self):
        while not self.closed:
            yield self.next_features

    def predict(self, feature_batch):
        """ Runs a prediction on a set of features. Calling multiple times
            does *not* regenerate the graph which makes predict much faster.

            feature_batch a list of list of features. IMPORTANT: If you're only classifying 1 thing,
            you still need to make it a batch of 1 by wrapping it in a list (i.e. predict([my_feature]), not predict(my_feature) 
        """
        self.next_features = feature_batch
        if self.first_run:
            self.batch_size = len(feature_batch)
            self.predictions = self.estimator.predict(
                input_fn=self.input_fn(self._create_generator))
            self.first_run = False
        elif self.batch_size != len(feature_batch):
            raise ValueError("All batches must be of the same size. First-batch:" + str(self.batch_size) + " This-batch:" + str(len(feature_batch)))

        results = []
        for _ in range(self.batch_size):
            results.append(next(self.predictions))
        return results

    def close(self):
        self.closed = True
        try:
            next(self.predictions)
        except:
            print("Exception in fast_predict. This is probably OK")


def example_input_fn(generator):
    """ An example input function to pass to predict. It must take a generator as input """

    def _inner_input_fn():
        dataset = tf.data.Dataset().from_generator(generator, output_types=(tf.float32)).batch(1)
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()
        return {'x': features}

    return _inner_input_fn
