from cifar_data import *
from classifier_nn import *

cfr = CifarData()
cfr.load_cifar10_data()
cfr.show_image()

nn = ClassifierNN(hidden_layers=[3072], input_data=cfr,
                  activation_func='sigmoid')

result = nn.train_network(batch_size=50, epochs=10,
                          train_func='gradientdescent', learning_rate=0.05)
