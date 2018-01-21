
from mxnet import gluon
from mxnet import symbol
import opt
from model.Hourglass import *
from model.Residual import *


def test_model(data):
    return createModel(data)

data = mx.symbol.Variable('data')
conv_comp= createModel(data)

shape= {"data" : (1,256,64,64)}
mx.viz.plot_network(symbol=conv_comp, shape=shape).view()
#
# data = mx.sym.var('data')
# # first conv layer
# conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
# tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
# pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
# # second conv layer
# conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
# tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
# pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
# # first fullc layer
# flatten = mx.sym.flatten(data=pool2)
# fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
# tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
# # second fullc
# fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
# # softmax loss
# lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
#
# # 可视化网络
# shape= {"data" : (1,3,256,256)}
# mx.viz.plot_network(symbol=lenet,shape=shape).view()