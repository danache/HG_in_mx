
import sys
print("______________--")
sys.path.insert(0, "/home/dan/HG_in_gulon/")
print(sys.path)
from dataprovider.dataProvider import hgIter
data_iter = hgIter(imgdir="/home/dan/ai_clg/", txt="/home/dan/ai_clg/a.txt",  resize=256, scale=0.25,outsize=64,normalize=True,flipping=False,color_jitting=30,mean_pixels=[0,0,0],
                 rotate=30, batch_size=1,  is_aug=False,randomize=True,joints_name=None,partnum=14,datasetname="train",isTraing=True

  # You can specify more augmentation options. Use help(mx.io.ImageRecordIter) to see all the options.
)
data,label = data_iter.next()