############stack parms #############
nStack = 3
nFeats = 256
nModules = 2
partnum = 14
############## train parms #########
learning_rate = 2.5e-4
momentum = 0.9
weight_decay = 0.0005
lr_refactor_step = '3,6,9,12,15,18'
lr_refactor_ratio = 0.5
############### base params #######
gpus = [0,1]
beginEpoch = 0
epoch = 20
freeze_pattern = ""
batch_size = 16
resume = -1
finetune = -1
prefix = "/media/bnrc2/_backup/model/mx/HG"

log_file = ""
log_frequent = 1
########## file params #################
train_img_path = "/media/bnrc2/_backup/ai/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/"
train_label = "/media/bnrc2/_backup/ai/ai_challenger_keypoint_train_20170902/keypoint_train_annotations_20170909.json"
train_file = "/media/bnrc2/_backup/dataset/aiclg/data.txt"

valid_img_path = "/media/bnrc2/_backup/ai/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911/"
valid_label = "/media/bnrc2/_backup/ai/ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json"
valid_file = "/media/bnrc2/_backup/dataset/aiclg/valid.txt"
