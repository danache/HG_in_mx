import mxnet as mx
import re
from dataprovider.dataProvider import hgIter
import opt
import os
from metric.MAPmetric import MapMetric
from model.Hourglass import createModel
import logging
def get_lr_scheduler(learning_rate, lr_refactor_step, lr_refactor_ratio,
                     num_example, batch_size, begin_epoch):
    """
    Compute learning rate and refactor scheduler

    Parameters:
    ---------
    learning_rate : float
        original learning rate
    lr_refactor_step : comma separated str
        epochs to change learning rate
    lr_refactor_ratio : float
        lr *= ratio at certain steps
    num_example : int
        number of training images, used to estimate the iterations given epochs
    batch_size : int
        training batch size
    begin_epoch : int
        starting epoch

    Returns:
    ---------
    (learning_rate, mx.lr_scheduler) as tuple
    """
    assert lr_refactor_ratio > 0
    iter_refactor = [int(r) for r in lr_refactor_step.split(',') if r.strip()]
    if lr_refactor_ratio >= 1:
        return (learning_rate, None)
    else:
        lr = learning_rate
        epoch_size = num_example // batch_size
        for s in iter_refactor:
            if begin_epoch >= s:
                lr *= lr_refactor_ratio
        if lr != learning_rate:
            logging.getLogger().info("Adjusted learning rate to {} for epoch {}".format(lr, begin_epoch))

        steps = [epoch_size * (x - begin_epoch) for x in iter_refactor if x > begin_epoch]
        if not steps:
            return (lr, None)
        print(steps)
        lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=lr_refactor_ratio)
        return (lr, lr_scheduler)
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if opt.log_file:
    fh = logging.FileHandler(opt.log_file)
    logger.addHandler(fh)

#######################define base params###############
ctx = [mx.gpu(int(i)) for i in opt.gpus]

####################### define network #################
symbol = createModel()
######################

epoch_size = opt.epoch
################### dataIter #########################
train =  hgIter(imgdir=opt.train_img_path, txt=opt.train_file,  resize=256, scale=0.25,outsize=64,normalize=True,
                flipping=False,color_jitting=30,mean_pixels=[0,0,0], rotate=30, batch_size=opt.batch_size,
                is_aug=False,randomize=True,joints_name=None,partnum=14,datasetname="train",isTraing=True)
###################
if opt.freeze_pattern.strip():
    re_prog = re.compile(opt.freeze_pattern)
    fixed_param_names = [name for name in symbol.list_arguments() if re_prog.match(name)]
else:
    fixed_param_names = None
ctx_str = '('+ ','.join([str(c) for c in ctx]) + ')'
resume = opt.resume
finetune = opt.finetune
prefix = opt.prefix
if resume > 0:
    logger.info("Resume training with {} from epoch {}"
        .format(ctx_str, resume))
    _, args, auxs = mx.model.load_checkpoint(prefix, resume)
    begin_epoch = resume
else:
    logger.info("Experimental: start training from scratch with {}"
        .format(ctx_str))
    args = None
    auxs = None
    fixed_param_names = None

if fixed_param_names:
    logger.info("Freezed parameters: [" + ','.join(fixed_param_names) + ']')

batch_end_callback = mx.callback.Speedometer(train.batch_size, frequent=opt.log_frequent)
epoch_end_callback = mx.callback.do_checkpoint(prefix,50)
num_example = train.getN()

mod = mx.mod.Module(symbol=symbol,
                context=ctx,
                data_names=['data'],
                label_names=['label'],
                    logger=logger)
# print(train.provide_data)
# print(train.provide_label)
mod.bind(data_shapes=train.provide_data, label_shapes=train.provide_label)
learning_rate, lr_scheduler = get_lr_scheduler(opt.learning_rate, opt.lr_refactor_step,
        opt.lr_refactor_ratio, num_example, opt.batch_size, opt.beginEpoch)

optimizer_params={'learning_rate':opt.learning_rate,
                  'gamma1':0.9,
                  'wd':opt.weight_decay,
                  'lr_scheduler':lr_scheduler,
                  'gamma2':0.9,
                  'clip_gradient':None,
                  'rescale_grad': 1.0 / len(ctx) if len(ctx) > 0 else 1.0 }

mod.fit(train,
        eval_metric=MapMetric(),
        # validation_metric=valid_metric,
        batch_end_callback=batch_end_callback,
        epoch_end_callback=epoch_end_callback,
        optimizer='RMSProp',
        optimizer_params=optimizer_params,
        begin_epoch=opt.beginEpoch,
        num_epoch=opt.epoch,
        initializer=mx.init.Xavier(),)
        # arg_params=args,
        # aux_params=auxs,
        # allow_missing=True,
        # monitor=monitor)