import mxnet as mx
from mxnet import gluon
# def convblock(data,numOut, name=""):
#     with mx.name.Prefix("%s_" % (name)):
#         bn1 = mx.symbol.BatchNorm(data=data, fix_gamma=False, eps=1e-5 + 1e-10, momentum=0.9, name='bn1')
#
#         relu1 = mx.symbol.Activation(data=bn1, act_type='relu', name='relu1' )
#
#         conv1 = mx.symbol.Convolution(data=relu1, num_filter=int(numOut / 2), kernel=(1,1),name='conv1' )
#
#         bn2 = mx.symbol.BatchNorm(data=conv1, fix_gamma=False, eps=1e-5 + 1e-10, momentum=0.9,
#                                   name='bn2' )
#         relu2 = mx.symbol.Activation(data=bn2, act_type='relu', name='relu2' )
#
#         conv2 = mx.symbol.Convolution(data=relu2, num_filter=int(numOut / 2), kernel=(3,3), stride=(1,1),
#                                       pad=(1,1), name='conv2' )
#         bn3 = mx.symbol.BatchNorm(data=conv2, fix_gamma=False, eps=1e-5 + 1e-10, momentum=0.9,
#                                   name='bn3' )
#         relu3 = mx.symbol.Activation(data=bn3, act_type='relu', name='relu3' )
#         conv3 = mx.symbol.Convolution(data=relu3, num_filter=numOut, kernel=(1,1), stride=(1,1),
#                                       name='conv3' )
#         return conv3
#
# def skipLayer(data,numin, numOut,name=""):
#     if numin == numOut:
#         return data
#     else:
#         conv = mx.symbol.Convolution(data=data, num_filter=numOut, kernel=(1,1), stride=(1,1),name='%s_conv'%(name) )
#         return conv
#
# def Residual(data,numin, numOut,name):
#     #with mx.name.Prefix("%s_%s_" % (name, suffix)):
#     convb = convblock(data, numOut,name="%s_convBlock" %(name))
#     skiplayer = skipLayer(data, numin,numOut, name="%s_skipLayer"%(name))
#     x = mx.symbol.add_n(convb, skiplayer,name="%s_add_layer"%(name))
#     return x

class convBlock(gluon.HybridBlock):
    def __init__(self, numOut ,**kwargs):
        super(convBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.bn1 = gluon.nn.BatchNorm(axis=1, center=True, scale=True)
            self.relu1 = gluon.nn.Activation(activation='relu')
            # conv1 + BN + Relu
            self.conv1 = gluon.nn.Conv2D(channels=int(numOut / 2), kernel_size=1, strides=1)
            self.bn2 = gluon.nn.BatchNorm(axis=1, center=True, scale=True)
            self.relu2 = gluon.nn.Activation(activation='relu')
            # conv2 + BN + Relu + Conv3
            self.conv2 = gluon.nn.Conv2D(channels=int(numOut / 2), kernel_size=3, strides=1, padding=1)
            self.bn3 = gluon.nn.BatchNorm(axis=1, center=True, scale=True)
            self.relu3 = gluon.nn.Activation(activation='relu')
            self.conv3 = gluon.nn.Conv2D(channels=numOut, kernel_size=1, strides=1)

    def hybrid_forward(self, F, x):
        # F is a function space that depends on the type of x
        # If x's type is NDArray, then F will be mxnet.nd
        # If x's type is Symbol, then F will be mxnet.sym
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x= self.bn2(x)
        x = self.relu2(x)
        # conv2 + BN + Relu + Conv3
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv3(x)
        return x






class skipLayer(gluon.HybridBlock):
    def __init__(self, numOut, **kwargs):
        super(skipLayer, self).__init__(**kwargs)
        with self.name_scope():
            self.numOut = numOut
            self.conv = gluon.nn.Conv2D(self.numOut,kernel_size=1,strides=1)

    def hybrid_forward(self, F, x):
        # F is a function space that depends on the type of x
        # If x's type is NDArray, then F will be mxnet.nd
        # If x's type is Symbol, then F will be mxnet.sym

        if x.shape[1] == self.numOut:
            return x
        else:
            with self.name_scope():
                x = self.conv(x)
                return x


class Residual(gluon.HybridBlock):
    def __init__(self, numOut, **kwargs):
        super(Residual, self).__init__(**kwargs)
        with self.name_scope():
            self.convB = convBlock(numOut)
            self.skip = skipLayer(numOut)
            self.register_child(self.convBlock)
            self.register_child(self.skipLayer)

    def hybrid_forward(self, F, x):
        # F is a function space that depends on the type of x
        # If x's type is NDArray, then F will be mxnet.nd
        # If x's type is Symbol, then F will be mxnet.sym

        x = F.add_n(self.convB(x), self.skip(x))

        return x

# def Residual(inputs, numOut,):
#     net = gluon.nn.Sequential()
#     with net.name_scope():
#         conv = convBlock(inputs, numOut)
#         skip = skipLayer(inputs, numOut,)
#         ########
#         print ("add_n test")
#         net.add(mx.symbol.add_n(conv, skip))
#     return net
