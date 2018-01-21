import mxnet as mx
from mxnet import gluon
from model.Residual import Residual
import opt
def hourglass(data,n, f,  nModual,name="",suffix=""):
    #with mx.name.Prefix("%s_%s_" % (name, suffix)):
    pool =  mx.symbol.Pooling(data=data, kernel=(2,2),stride=(2,2),pool_type="max",name='%s_pool1' %(name) )

    up = []
    low = []

    for i in range(nModual):

        tmpup = None
        tmplow = None
        if i == 0:

            tmpup = Residual(data,f,f,name='%s_tmpup_'%(name) + str(i))
            tmplow = Residual(pool,f,f,name='%s_tmplow_'%(name) + str(i))
        else:

            tmpup =  Residual(up[i-1],f,f,name='%s_tmpup_'%(name) + str(i))
            tmplow = Residual(low[i - 1],f,f,name='%s_tmplow_'%(name) + str(i))

        up.append(tmpup)
        low.append(tmplow)

    ####################
    #####################

    low2 = []
    if n > 1:
        low2.append(hourglass(low[-1],n - 1,f,nModual=nModual,name=name+"_" + str(n - 1)+"_low2"))
    else:
        for j in range(nModual):
            if j == 0:
                tmp_low2 = Residual(low[-1], f, f, name='%s_tmplow2_' % (name) + str(j))
            else:
                tmp_low2 = Residual(low2[j - 1], f,f,name=name+"_"+str(n - 1)+"_low2 %d" %j)
            low2.append(tmp_low2)

    low3_ = []
    for k in range(nModual):
        if k == 0:
            tmplow3 = Residual(low2[-1], f, f, name='%s_tmplow3_' % (name) + str(k))
        else:
            tmplow3 = Residual(low3_[k - 1], f, f, name='%s_tmplow3_' % (name) + str(k))
        low3_.append(tmplow3)

    up2 = mx.symbol.UpSampling(low3_[-1], scale=2, sample_type='nearest',name="%s_Upsample"%(name))


    comb = mx.symbol.add_n(up[nModual - 1], up2,name=name+"_add")
    return comb


def lin(data, numOut,name=None, suffix=""):
    with mx.name.Prefix("%s_%s_" % (name, suffix)):

        conv1 = mx.symbol.Convolution(data=data, num_filter=numOut, kernel=(1,1), stride=(1,1),
                                      name='conv1' )
        bn1 = mx.symbol.BatchNorm(data=conv1, fix_gamma=False, eps=1e-5 + 1e-10, momentum=0.9,
                                  name='bn1' )

        relu1 = mx.symbol.Activation(data=bn1, act_type='relu', name='relu1' )
        return relu1

def createModel():
    data = mx.sym.Variable(name='data')
    label = mx.sym.Variable(name='label')
    # label = mx.symbol.Variable(name="hg_label")
    conv1 = mx.symbol.Convolution(data=data, num_filter=64, kernel=(7,7), stride=(2,2), pad=(3,3),name="conv1")


    bn1 = mx.symbol.BatchNorm(data=conv1, fix_gamma=False, eps=1e-5 + 1e-10, momentum=0.9,name="bn1")

    relu1 = mx.symbol.Activation(data=bn1, act_type='relu',name="relu1")
    r1 = Residual(relu1, 64,128,name="Residual1")

    pool1 = mx.symbol.Pooling(data=r1, kernel=(2,2),stride=(2,2),pool_type="max",name="pool1")
    r2 = Residual(pool1, 128,128,name="Residual2")

    r3 = Residual(r2, 128,opt.nFeats,name="Residual3")
    # return r3
    #r3 = data
    # pool2 = mx.symbol.Pooling(data=r3, kernel=(2,2),stride=(2,2), pool_type="max",name="pool2")
    # r4 = Residual(pool2, 128,128,name="Residual4")
    #
    # r5 = Residual(r4, 128,128,name="Residual5")
    #
    # r6 = Residual(r5,128,opt.nFeats,name="Residual6")
    ####################################################
    #r6 = data
    ####################################################

    ###################################################test

    hg = [None] * opt.nStack

    ll = [None] * opt.nStack
    fc_out = [None] * opt.nStack
    c_1 = [None] * opt.nStack
    c_2 = [None] * opt.nStack
    sum_ = [None] * opt.nStack
    resid = dict()
    out = []



    hg[0] = hourglass(r3, n=4, f=opt.nFeats, name="stage_0_hg", nModual=opt.nModules)

    resid["stage_0"] = []
    for i in range(opt.nModules):
        if i == 0:
            tmpres = Residual(hg[0], opt.nFeats, opt.nFeats, name='stage_0tmpres_%d' % (i))
        else:
            tmpres = Residual(resid["stage_0"][i - 1], opt.nFeats, opt.nFeats, name='stage_0tmpres_%d' % (i),
                              )
        resid["stage_0"].append(tmpres)

    ll[0] = lin(resid["stage_0"][-1], opt.nFeats, name="stage_0_lin1")
    fc_out[0] = mx.symbol.Convolution(data=ll[0], num_filter=opt.partnum, kernel=(1, 1),
                        name="stage_0_out")
    
    out.append(fc_out[0])
    if opt.nStack > 1:
        c_1[0] = mx.symbol.Convolution(ll[0], num_filter=opt.nFeats, kernel=(1, 1),
                         name="stage_0_conv1")

        c_2[0] = mx.symbol.Convolution(c_1[0], num_filter=opt.nFeats, kernel=(1, 1),
                         name="stage_0_conv2")
        sum_[0] = mx.symbol.add_n(r3, c_1[0], c_2[0],name="stage_0_add_n")
    ####stage 2 - n-1

    for i in range(1, opt.nStack - 1):


        hg[i] = hourglass(sum_[i - 1], n=4, f=opt.nFeats, nModual=opt.nModules,
                               name="stage_%d_hg" % (i))

        resid["stage_%d" % i] = []
        for j in range(opt.nModules):
            if j == 0:
                tmpres = Residual(hg[i], opt.nFeats, opt.nFeats, name='stage_%d_tmpres_%d' % (i, j),
                                 )
            else:
                tmpres = Residual(resid["stage_%d" % i][j - 1], opt.nFeats, opt.nFeats,
                                  name='stage_%d_tmpres_%d' % (i, j),
                                  )
            resid["stage_%d" % i].append(tmpres)

        ll[i] = lin(resid["stage_%d" % i][-1], opt.nFeats, name="stage_%d_lin" % (i))
        fc_out[i] = mx.symbol.Convolution(ll[i],num_filter= opt.partnum, kernel=(1, 1),
                            name="stage_%d_out" % (i))
        out.append(fc_out[i])

        c_1[i] = mx.symbol.Convolution(ll[i], num_filter=opt.nFeats, kernel=(1, 1),
                         name="stage_%d_conv1" % (i))

        c_2[i] = mx.symbol.Convolution(c_1[i], num_filter=opt.nFeats, kernel=(1, 1),
                         name="stage_%d_conv2" % (i))
        sum_[i] = mx.symbol.add_n(sum_[i - 1], c_1[i], c_2[i],name="stage_%d_add_n" % (i))
        ###stage end
    hg[opt.nStack - 1] = hourglass(sum_[opt.nStack - 2], n=4, f=opt.nFeats, nModual=opt.nModules,
                                         name="stage_%d_hg" % (opt.nStack - 1))
    residual = []
    for j in range(opt.nModules):
        if j == 0:
            tmpres = Residual(hg[opt.nStack - 1], opt.nFeats, opt.nFeats,
                              name='stage_%d_tmpres_%d' % (opt.nStack - 1, j)
                              )
        else:
            tmpres = Residual(residual[j - 1], opt.nFeats, opt.nFeats,
                              name='stage_%d_tmpres_%d' % (opt.nStack - 1, j)
                              )
        residual.append(tmpres)

    ll[opt.nStack - 1] = lin(residual[-1], opt.nFeats,
                                   name="stage_%d_lin1" % (opt.nStack - 1))
    fc_out[opt.nStack - 1] = mx.symbol.Convolution(ll[opt.nStack - 1], num_filter=opt.partnum, kernel=(1, 1),

                                      name="stage_%d_out" % (opt.nStack - 1))
    out.append(fc_out[opt.nStack - 1])
    # end = out[0]
    # end = tl.layers.StackLayer(out, axis=1, name='gfinal_output')
    allloss = []
    for i in range(opt.nStack):
        allloss.append(mx.sym.LinearRegressionOutput(data=out[i],label=label))
    loss = mx.sym.add_n(*allloss)
    loss = mx.symbol.make_loss(loss,name="loss")
    return loss

    # ##########################################################3
    # for i in range(opt.nStack):
    #
    #     hg = hourglass(data=inter[i],n=nPool,f=opt.nFeats,imsize=opt.outputRes,nModual=int(nModual),name="hourglass%d" %(i))#n=nPool,f=opt.nFeats,imsize=opt.outputRes,nModual=int(nModual))
    #     ll1 = None
    #     ll2 = None
    #     att = None
    #     tmpOut = None
    #     if i == opt.nStack - 1:
    #         ll1 = lin(hg,opt.nFeats * 2,name="hourglass%d_lin1"%(i))
    #         ll2 = lin(ll1,opt.nFeats * 2,name="hourglass%d_lin2"%(i))
    #         att = AttentionPartsCRF(ll2, opt.nFeats * 2,opt.LRNKer, 3, 0,name="hourglass%d_attention1"%(i))
    #         tmpOut = AttentionPartsCRF(att, opt.nFeats * 2,opt.LRNKer, 3, 1,name="hourglass%d_attention2"%(i))
    #     else:
    #         ll1 = lin(hg, opt.nFeats,name="hourglass%d_lin1"%(i))
    #         ll2 = lin(ll1, opt.nFeats,name="hourglass%d_lin2"%(i))
    #
    #         if i >= 4 :
    #             att = AttentionPartsCRF(ll2, opt.nFeats,opt.LRNKer, 3, 0,name="hourglass%d_attention1"%(i))
    #             tmpOut = AttentionPartsCRF(att, opt.nFeats,opt.LRNKer, 3, 1,name="hourglass%d_attention2"%(i))
    #         else:
    #
    #             att = AttentionPartsCRF(ll2, opt.nFeats,opt.LRNKer, 3, 0,name="hourglass%d_attention1"%(i))
    #
    #             tmpOut = mx.symbol.Convolution(data=att, num_filter=opt.partnum, kernel=(1,1), stride=(1,1), pad=0,name="hourglass%d_tmpout"%(i))
    #
    #     out.append(tmpOut)
    #
    #     if i < opt.nStack - 1:
    #         outmap = mx.symbol.Convolution(data=tmpOut, num_filter=256, kernel=(1,1), stride=(1,1), pad=0,name="hourglass%d_conv"%(i))
    #         ll3 = lin(outmap, opt.nFeats,name="hourglass%d_lin3"%(i))
    #         toointer = mx.symbol.add_n(inter[i],outmap,ll3,name="add_n%d"%(i))
    #         inter.append(toointer)
    #
    # arg_shapes, out_shapes, aux_shapes = out[0].infer_shape(data=(1, 3, 256, 256))
    # print(out_shapes)
    #
    #
    # for i in range(len(out)):
    #     out[i] = mx.symbol.expand_dims(out[i],axis=1)
    # loss = mx.symbol.concat(*out,dim=1 ,name="loss")
    # arg_shapes, out_shapes, aux_shapes = loss.infer_shape(data=(1, 3, 256, 256))
    # print(out_shapes)
    # #return out[opt.nStack - 1]
    # #group = mx.sym.Group(out)
    # loss = mx.symbol.LinearRegressionOutput(data=loss,label=label,name="hg")
    #
    # return loss
    #
