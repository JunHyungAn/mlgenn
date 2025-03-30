from tensorflow.keras.datasets import mnist
import numpy as np
import sys
sys.path.append("C:/Users/wnswk/ml_genn")
from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.compilers import InferenceCompiler
from ml_genn.neurons import IntegrateFire, IntegrateFireInput
from ml_genn.connectivity import Dense
from ml_genn.callbacks import VarRecorder
from time import perf_counter

BATCH_SIZE = 128

# Create sequential model
network = SequentialNetwork()
with network:
    input = InputLayer(IntegrateFireInput(v_thresh=5.0), 784)
    Layer(Dense(weight=np.load("C:/Users/wnswk/ml_genn/examples/weights_0_1.npy")),
          IntegrateFire(v_thresh=5.0))
    output = Layer(Dense(weight=np.load("C:/Users/wnswk/ml_genn/examples/weights_1_2.npy")),
                   IntegrateFire(v_thresh=5.0, readout="spike_count"))

compiler = InferenceCompiler(dt=1.0, batch_size=BATCH_SIZE,
                             evaluate_timesteps=100)
compiled_net = compiler.compile(network)

# Load testing data
# mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
# testing_images = np.reshape(mnist.test_images(), (-1, 784))
# testing_labels = mnist.test_labels()
(X_train, y_train), (X_test, y_test) = mnist.load_data()

testing_images = X_test.astype("float32") / 255.0
testing_images = testing_images.reshape((-1, 784))
testing_labels = y_test

with compiled_net:
    # Evaluate model on numpy dataset
    start_time = perf_counter()
    metrics, _ = compiled_net.evaluate({input: testing_images * 0.01},
                                       {output: testing_labels})
    end_time = perf_counter()
    print(f"Accuracy = {100 * metrics[output].result}%")
    print(f"Time = {end_time - start_time}s")