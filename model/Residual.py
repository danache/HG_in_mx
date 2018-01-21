import mxnet as mx
from mxnet import gluon
def convblock(data,numOut, name=""):
    with mx.name.Prefix("%s_" % (name)):
        bn1 = mx.symbol.BatchNorm(data=data, fix_gamma=False, eps=1e-5 + 1e-10, momentum=0.9, name='bn1')

        relu1 = mx.symbol.Activation(data=bn1, act_type='relu', name='relu1' )

        conv1 = mx.symbol.Convolution(data=relu1, num_filter=int(numOut / 2), kernel=(1,1),name='conv1' )

        bn2 = mx.symbol.BatchNorm(data=conv1, fix_gamma=False, eps=1e-5 + 1e-10, momentum=0.9,
                                  name='bn2' )
        relu2 = mx.symbol.Activation(data=bn2, act_type='relu', name='relu2' )

        conv2 = mx.symbol.Convolution(data=relu2, num_filter=int(numOut / 2), kernel=(3,3), stride=(1,1),
                                      pad=(1,1), name='conv2' )
        bn3 = mx.symbol.BatchNorm(data=conv2, fix_gamma=False, eps=1e-5 + 1e-10, momentum=0.9,
                                  name='bn3' )
        relu3 = mx.symbol.Activation(data=bn3, act_type='relu', name='relu3' )
        conv3 = mx.symbol.Convolution(data=relu3, num_filter=numOut, kernel=(1,1), stride=(1,1),
                                      name='conv3' )
        return conv3

def skipLayer(data,numin, numOut,name=""):
    if numin == numOut:
        return data
    else:
        conv = mx.symbol.Convolution(data=data, num_filter=numOut, kernel=(1,1), stride=(1,1),name='%s_conv'%(name) )
        return conv

def Residual(data,numin, numOut,name):
    #with mx.name.Prefix("%s_%s_" % (name, suffix)):
    convb = convblock(data, numOut,name="%s_convBlock" %(name))
    skiplayer = skipLayer(data, numin,numOut, name="%s_skipLayer"%(name))
    x = mx.symbol.add_n(convb, skiplayer,name="%s_add_layer"%(name))
    return x