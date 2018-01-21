import mxnet as mx
from mxnet import gluon
from model.Residual import Residual
class hourglass(gluon.HybridBlock):
    def __init__(self, n, output_shape, imsize, nModual, **kwargs):
        super(hourglass, self).__init__(**kwargs)
        with self.name_scope():
            self.nModual = nModual
            self.pool = gluon.nn.MaxPool2D(pool_size=2,strides=2)
            self.up = []
            self.low = []
            self.low2_lst =[]
            self.low3_lst = []
            self.n = n
            for i in range(int(nModual)):

                tmpup = Residual(output_shape)
                tmplow = Residual(output_shape)
                self.register_child(tmpup)
                self.register_child(tmplow)
                self.up.append(tmpup)
                self.up.append(tmplow)

            if n > 1:
                low2 = hourglass(n-1,output_shape,imsize/2,nModual)
                self.register_child(low2)
                self.low2_lst.append(low2)
            else:
                for j in range(int(nModual)):
                    low2 = Residual(output_shape)
                    self.register_child(low2)
                    self.low2_lst.append(low2)
            for k in range(int(nModual)):
                low3 = Residual(output_shape)
                self.register_child(low3)

                self.low3_lst.append(low3)

    def hybrid_forward(self, F, x):
        # F is a function space that depends on the type of x
        # If x's type is NDArray, then F will be mxnet.nd
        # If x's type is Symbol, then F will be mxnet.sym
        pool = self.pool(x)
        for i in range(int(self.nModual)):
            x = self.up[i](x)
            pool = self.low[i](x)
        if self.n > 1:
            pool = self.low2_lst[-1](pool)
        else:
            for j in range(int(self.nModual)):
                pool = self.low2_lst[j](pool)
        for k in range(int(self.nModual)):

            pool = self.low3_lst[k](pool)

        up2 =  F.UpSampling(pool, scale=2, sample_type='nearest')
        comb = F.add_n(x, up2)
        return comb

class lin(gluon.HybridBlock):
    def __init__(self,  numOut,  **kwargs):
        super(lin, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = gluon.nn.Conv2D(channels=numOut,kernel_size=1,strides=1,padding=0)
            self.bn1 = gluon.nn.BatchNorm(axis=1, center=True, scale=True)
            self.relu1 = gluon.nn.Activation(activation='relu')

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x

class createModel(gluon.HybridBlock):
    def __init__(self,  nPool, nStack,nFeats,outputRes,LRNKer, **kwargs):
        super(createModel, self).__init__(**kwargs)
        self.nStack = nStack
        with self.name_scope():
            self.cnv1_ = gluon.nn.Conv2D(channels=64, kernel_size=7, strides=1, padding=3)
            # cnv1
            self.cnv1 = gluon.nn.BatchNorm(axis=1, center=True, scale=True)
            # relu1
            self.relu1 = gluon.nn.Activation(activation='relu')
            # r1
            self.r1 = Residual(64)
            # pool1
            self.pool1 = gluon.nn.MaxPool2D(pool_size=2, strides=2, )
            # r2
            self.r2 = Residual(64)
            # r3
            self.r3 = Residual(128)

            # pool2
            self.pool2 = gluon.nn.MaxPool2D(pool_size=2, strides=2, )
            # r4
            self.r4 = Residual(128)
            # r5
            self.r5 = Residual(128)
            # r6
            self.r6 = Residual(nFeats)

            self.out = []

            self.ll1_set = []
            self.ll2_set= []
            self.att1_set = []

            self.hg_set = []

            self.outmap_set = []
            self.ll3_set = []

            if nPool == 3:
                nModual = 16 / nStack
            else:
                nModual = 8 / nStack
            for i in range(nStack):
                hg = hourglass(nPool, nFeats, outputRes, nModual)
                self.register_child(hg)
                self.hg_set.append(hg)
                ll1 = None
                ll2 = None
                att = None
                tmpout = None
                if i == nStack - 1:
                    ll1 = lin(nFeats * 2)
                    ll2 = lin(nFeats * 2)
                    att = AttentionPartsCRF(LRNKer,3,0)
                    tmpout = AttentionPartsCRF(LRNKer,3,1)
                else:
                    ll1 = lin(nFeats)
                    ll2 = lin(nFeats)
                    if i >= 4:
                        att = AttentionPartsCRF(LRNKer, 3, 0)
                        tmpout = AttentionPartsCRF(LRNKer, 3, 1)
                    else:
                        att = AttentionPartsCRF(LRNKer, 3, 0)
                        tmpout = gluon.nn.Conv2D(channels=14, kernel_size=1, strides=1, padding=0)
                self.register_child(ll1)
                self.register_child(ll2)
                self.register_child(att)
                self.register_child(tmpout)

                self.ll1_set.append(ll1)
                self.ll2_set.append(ll2)
                self.att1_set.append(att)
                self.out.append(tmpout)


                if i < nStack - 1:
                    outmap = gluon.nn.Conv2D(256, strides=1, kernel_size=1, padding=0)
                    ll3 = lin(nFeats)
                    self.register_child(outmap)
                    self.register_child(ll3)
                    self.outmap_set.append(outmap)
                    self.ll3_set.append(ll3)

    def hybrid_forward(self, F, x):
        x = self.cnv1_(x)
        x = self.cnv1(x)
        x = self.r1(x)
        x = self.pool1(x)
        x = self.r2(x)
        x = self.r3(x)
        x = self.pool2(x)
        x = self.r4(x)
        x = self.r5(x)
        x = self.r6(x)
        inter = x
        for i in range(self.nStack):
            hg = self.hg_set[i](x)
            ll1 = self.ll1_set[i](hg)
            ll2 = self.ll2_set[i](ll1)
            att = self.att1_set[i](ll2)
            tmpout = self.out[i](att)
            if i < self.nStack - 1 :
                outmap = self.outmap_set[i](tmpout)
                ll3 = self.ll3_set[i](ll1)
                tmpout = F.add_n(inter,outmap,ll3)
                inter = tmpout
            x = tmpout

        return x
