from cifar_data import *
from classifier_nn import *

cfr = CifarData()
cfr.load_cifar10_data()
cfr.show_image()

nn = ClassifierNN(hidden_layers=[500, 500], input_data=cfr,
                  activation_func='relu')

result = nn.train_network(epochs=5)
