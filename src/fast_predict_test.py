from fast_predict import FastPredict
import tensorflow as tf
import numpy as np

# Train a very simple model.
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=1)]
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10],
                                            n_classes=2,
                                            model_dir="/tmp/fast_predict_test_model/")


sample = np.array([5], dtype=np.float32)
label = np.array([1], dtype=np.float32)
classifier.fit(x = sample,
               y = label,
               steps = 1)


# Verify that the model isn't being re-loaded for each predict call (look for log lines like "Creating TensorFlow device...")
fast_predict = FastPredict(classifier)
for i in range(5):
    print(fast_predict.predict(np.array([i], dtype=np.float32)))
    
