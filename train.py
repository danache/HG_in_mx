import mxnet as mx
from dataprovider.dataProvider import hgIter
import opt
import os
from model.Hourglass import createModel
# kv = mx.kvstore.create(opt.kv_store)
# devs = mx.cpu() if opt.gpus is None else [mx.gpu(int(i)) for i in opt.gpus.split(',')]
# epoch_size = max(int(opt.num_examples / opt.batch_size / kv.num_workers), 1)
# begin_epoch = opt.model_load_epoch if opt.model_load_epoch else 0
# if not os.path.exists("./model"):
#     os.mkdir("./model")
# model_prefix = "model/resnet-{}-{}-{}".format(opt.data_type, opt.depth, kv.rank)
# checkpoint = mx.callback.do_checkpoint(model_prefix)
ctx = [mx.gpu(0)]
begin_epoch = 0
end_epoch = 0

arg_params = None
aux_params = None
symbol = createModel()
# if opt.retrain:
#     _, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, opt.model_load_epoch)
# if opt.memonger:
#     import memonger
#     symbol = memonger.search_plan(symbol, data=(opt.batch_size, 3, 32, 32) if opt.data_type=="cifar10"
#                                                 else (opt.batch_size, 3, 224, 224))
epoch_size = 200
train =  hgIter(imgdir="/home/dan/ai_clg/", txt="/home/dan/ai_clg/a.txt",  resize=256, scale=0.25,outsize=64,normalize=True,flipping=False,color_jitting=30,mean_pixels=[0,0,0],
                 rotate=30, batch_size=1,  is_aug=False,randomize=True,joints_name=None,partnum=14,datasetname="train",isTraing=True

  # You can specify more augmentation options. Use help(mx.io.ImageRecordIter) to see all the options.
)
def multi_factor_scheduler(begin_epoch, epoch_size, step=[60, 75, 90], factor=0.1):
    step_ = [epoch_size * (x-begin_epoch) for x in step if x-begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None

# train = mx.io.ImageRecordIter(
#     path_imgrec         = os.path.join(opt.data_dir, "cifar10_train.rec") if opt.data_type == 'cifar10' else
#                           os.path.join(opt.data_dir, "train_256_q90.rec") if opt.aug_level == 1
#                           else os.path.join(opt.data_dir, "train_480_q90.rec"),
#     label_width         = 1,
#     data_name           = 'data',
#     label_name          = 'softmax_label',
#     data_shape          = (3, 32, 32) if opt.data_type=="cifar10" else (3, 224, 224),
#     batch_size          = opt.batch_size,
#     pad                 = 4 if opt.data_type == "cifar10" else 0,
#     fill_value          = 127,  # only used when pad is valid
#     rand_crop           = True,
#     max_random_scale    = 1.0,  # 480 with imagnet, 32 with cifar10
#     min_random_scale    = 1.0 if opt.data_type == "cifar10" else 1.0 if opt.aug_level == 1 else 0.533,  # 256.0/480.0
#     max_aspect_ratio    = 0 if opt.data_type == "cifar10" else 0 if opt.aug_level == 1 else 0.25,
#     random_h            = 0 if opt.data_type == "cifar10" else 0 if opt.aug_level == 1 else 36,  # 0.4*90
#     random_s            = 0 if opt.data_type == "cifar10" else 0 if opt.aug_level == 1 else 50,  # 0.4*127
#     random_l            = 0 if opt.data_type == "cifar10" else 0 if opt.aug_level == 1 else 50,  # 0.4*127
#     max_rotate_angle    = 0 if opt.aug_level <= 2 else 10,
#     max_shear_ratio     = 0 if opt.aug_level <= 2 else 0.1,
#     rand_mirror         = True,
#     shuffle             = True,
#     num_parts           = kv.num_workers,
#     part_index          = kv.rank)
# val = mx.io.ImageRecordIter(
#     path_imgrec         = os.path.join(opt.data_dir, "cifar10_val.rec") if opt.data_type == 'cifar10' else
#                           os.path.join(opt.data_dir, "val_256_q90.rec"),
#     label_width         = 1,
#     data_name           = 'data',
#     label_name          = 'softmax_label',
#     batch_size          = opt.batch_size,
#     data_shape          = (3, 32, 32) if opt.data_type=="cifar10" else (3, 224, 224),
#     rand_crop           = False,
#     rand_mirror         = False,
#     num_parts           = kv.num_workers,
#     part_index          = kv.rank)
mod = mx.mod.Module(symbol=symbol,
                context=mx.gpu(0),
                data_names=['data'],
                label_names=['label'])
#mod.bind(data_shapes=[('data', (1, 3, 256, 256))], for_training=False)
mod.bind(data_shapes=train.provide_data, label_shapes=train.provide_label)
# batch_end_callback = mx.callback.Speedometer(train.batch_size, frequent=frequent)
# epoch_end_callback = mx.callback.do_checkpoint(prefix)
# learning_rate, lr_scheduler = get_lr_scheduler(learning_rate, lr_refactor_step,
#     lr_refactor_ratio, num_example, batch_size, begin_epoch)
# optimizer_params={'learning_rate':learning_rate,
#                   'momentum':momentum,
#                   'wd':weight_decay,
#                   'lr_scheduler':lr_scheduler,
#                   'clip_gradient':None,
#                   'rescale_grad': 1.0 / len(ctx) if len(ctx) > 0 else 1.0 }
# monitor = mx.mon.Monitor(iter_monitor, pattern=monitor_pattern) if iter_monitor > 0 else None

# run fit net, every n epochs we run evaluation network to get mAP
# if voc07_metric:
#     valid_metric = VOC07MApMetric(ovp_thresh, use_difficult, class_names, pred_idx=3)
# else:
#     valid_metric = MApMetric(ovp_thresh, use_difficult, class_names, pred_idx=3)

optimizer_params={'learning_rate':opt.learning_rate,
                      'momentum':opt.momentum,
                      'wd':opt.weight_decay,
                      'lr_scheduler':opt.lr_scheduler,
                      'clip_gradient':None,
                      'rescale_grad': 1.0 / len(ctx) if len(ctx) > 0 else 1.0 }
print("model fit")
mod.fit(train,

        # eval_metric=MultiBoxMetric(),
        # validation_metric=valid_metric,
        # batch_end_callback=batch_end_callback,
        # epoch_end_callback=epoch_end_callback,
        optimizer='sgd',
        optimizer_params=optimizer_params,
        begin_epoch=begin_epoch,
        num_epoch=end_epoch,
        initializer=mx.init.Xavier(),)
        # arg_params=args,
        # aux_params=auxs,
        # allow_missing=True,
        # monitor=monitor)