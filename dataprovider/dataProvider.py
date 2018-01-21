import mxnet as mx
import numpy as np
from mxnet.io import DataIter, DataBatch
import random
from matplotlib import pyplot as plt
import opt
import os
import scipy
class hgIter(mx.io.DataIter):
    """
    user for generate iter including heatmap
    """
    def __init__(self, imgdir=None, txt=None,  resize=256, scale=0.25,outsize=64,normalize=True,flipping=False,color_jitting=30,mean_pixels=[0,0,0],
                 rotate=30, batch_size=32,  is_aug=False,randomize=True,joints_name=None,partnum=14,datasetname="train",isTraing=True):

        self.joints_name = joints_name
        self.letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        self.img_dir = imgdir
        self.train_data_file = txt
        self.images = os.listdir(imgdir)
        self.israndom = randomize
        self.name = datasetname
        self.normalize = normalize
        self.batch_size = batch_size
        self.joints_name = joints_name
        if is_aug:
            self.flipping = flipping
            self.color_jitting = color_jitting
            self.rotate = rotate
        else:
            self.flipping = False
            self.color_jitting = False
            self.rotate = False
        self.img_size = resize
        self.hm_size = outsize
        self.isTrainging = isTraing
        self.outsize=outsize
        self.data = mx.nd.zeros((self.batch_size, 3, resize, resize))
        self.partnum = partnum
        self.label = mx.nd.zeros((self.batch_size, outsize ,outsize,self.partnum))
        self.cursor = -1
        self.creatset()
    def creatset(self):
        self.load_data()
        if self.israndom:
            self._randomize()
        self.dataset = self.train_table
        self.num_data = len(self.dataset)
        print('SET CREATED')
        np.save('Dataset% s'%self.name, self.dataset)
        print('--Training set :', self.num_data, ' samples.')

    def load_data(self):
        self.train_table = []
        self.no_intel = []
        self.data_dict = {}
        input_file = open(self.train_data_file, 'r')
        print('READING TRAIN DATA')
        for line in input_file:
            line = line.strip()
            line = line.split(' ')
            name = line[0]
            box = list(map(int, line[1:5]))
            joints = list(map(int, line[5:]))
            if joints == [-1] * len(joints):
                self.no_intel.append(name)
            else:
                joints = np.reshape(joints, (-1, 2))
                w = [1] * joints.shape[0]
                for i in range(joints.shape[0]):
                    if np.array_equal(joints[i], [-1, -1]):
                        w[i] = 0
                self.data_dict[name] = {'box': box, 'joints': joints, 'weights': w}
                self.train_table.append(name)
        input_file.close()

    def _randomize(self):
        """ Randomize the set
        """
        random.shuffle(self.train_table)

    def open_img(self, name, color='RGB'):
        """ Open an image
        Args:
            name	: Name of the sample
            color	: Color Mode (RGB/BGR/GRAY)
        """
        if name[-1] in self.letter:
            name = name[:-1]

        img_name =os.path.join(self.img_dir, name)
        img = scipy.misc.imread(img_name)
        return img

    # def __init__(self,path_imgrec,data_shape,batch_size,max_rotate_angle,mean_pixels=[0,0,0],rand_mirror=True,color_jitter=20):
    #     super(hgIter, self).__init__()
    #     self.rec = mx.io.ImageRecordIter(
    #         path_imgrec=path_imgrec,  # The target record file.
    #         # Output data shape; 227x227 region will be cropped from the original image.
    #         label_width=46,
    #         data_shape=data_shape,
    #         batch_size=batch_size,  # Number of items per batch.
    #         #rand_mirror=rand_mirror,
    #         random_h=color_jitter,
    #         random_s=color_jitter,
    #         random_l=color_jitter,
    #         mean_r = mean_pixels[0],
    #         mean_g=mean_pixels[1],
    #         mean_b=mean_pixels[2],
    #
    #         # You can specify more augmentation options. Use help(mx.io.ImageRecordIter) to see all the options.
    #     )
    #     self.batch_size = batch_size
    #     self._get_batch()
    #     if not self.provide_label:
    #         raise RuntimeError("Invalid ImageDetRecordIter: " + path_imgrec)
    #     self.reset()


    def _get_batch(self):
        self._batch = self.rec.next()
        if not self._batch:
            return False
        #data = mx.nd.cast(self._batch.data[0],dtype='uint8').asnumpy()
        #data = np.transpose(data,(2,3,1,0))
        #img_data = np.reshape(data, (256,256,3))
        #img_data = cv2.fromarray(img_data)
        #imgcp = img_data.copy()
        heatmap = np.zeros((self._batch.label[0].shape[0],opt.partnum,opt.outputRes, opt.outputRes ), dtype=np.float32)

        for i in range(self._batch.label[0].shape[0]):
            label = self._batch.label[0][i].asnumpy()
            for j in range(opt.partnum):
                x = 4 + j * 3 + 1
                y = 4 + j * 3 + 2
                #imgcp = cv2.circle(imgcp, (int(label[x]*256), int(label[y]*256)), 15, (0, 0, 255), -1)
                s = int(np.sqrt(opt.outputRes) * opt.outputRes * 10 / 4096) + 2
                hm = self._makeGaussian(opt.outputRes, opt.outputRes, sigma=s, center=(label[x]*opt.outputRes, label[y]*opt.outputRes))
                heatmap[i,j,:, :] = hm
                #hm = hm.astype(np.uint8)

        y_batch = np.zeros((self.batch_size, opt.nStack, opt.partnum,opt.outputRes, opt.outputRes ))
        for i in range(opt.nStack):
            y_batch[:,i,:,:,:] = heatmap
        self.label_shape = (self.batch_size, opt.nStack, opt.partnum, opt.outputRes, opt.outputRes)
        self.provide_label = [('hg_label', self.label_shape)]
        #cv2.imwrite("/home/dan/test_img/2222.jpg", imgcp)
        self._batch.label = [mx.nd.array(y_batch)]
        #self.showHeatMap()
        return True

    def showHeatMap(self):
        for i in range(self.heatmap.shape[0]):
            fig = plt.figure()
            for j in range(14):
                ax = fig.add_subplot(4,4,j+1)
                img = self.heatmap[i,:,:,j]
                ax.imshow(img)
            plt.show()

    # def _augment(self, img, hm, max_rotation=30):
    #     """ # TODO : IMPLEMENT DATA AUGMENTATION
    #     """
    #     if random.choice([1]):
    #         r_angle = np.random.randint(-1 * max_rotation, max_rotation)
    #         img = transform.rotate(img, r_angle, preserve_range=True)
    #         hm = transform.rotate(hm, r_angle)
    #     return img, hm


                # img = Image.fromarray(data)
        # img.save("/home/dan/text_img/ssss.png")
        #for i in range(14):

        #self.heatmap = self._generate_hm(256, 256, 14, 256)

    def get_transform(self,center, scale, res, rot=0):
        # Generate transformation matrix
        h = scale
        t = np.zeros((3, 3))
        t[0, 0] = float(res[1]) / (h[1])
        t[1, 1] = float(res[0]) / (h[0])
        t[0, 2] = res[1] * (-float(center[0]) / h[1] + .5)
        t[1, 2] = res[0] * (-float(center[1]) / h[0] + .5)
        t[2, 2] = 1
        return t

    def getN(self):
        return len(self.dataset)

    def transform(self,pt, center, scale, res, invert=0, rot=0):
        # Transform pixel location to different reference
        t = self.get_transform(center, scale, res, rot=rot)
        if invert:
            t = np.linalg.inv(t)
        new_pt = np.array([pt[0], pt[1], 1.]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2].astype(int)

    def transformPreds(self,coords, center, scale, res, reverse=0):
        #     local origDims = coords:size()
        #     coords = coords:view(-1,2)
        lst = []

        for i in range(coords.shape[0]):
            lst.append(self.transform(coords[i], center, scale, res, reverse, 0, ))

        newCoords = np.stack(lst, axis=0)

        return newCoords

    def crop(self,img, center, scale, res):

        ul = np.array(self.transform([0, 0], center, scale, res, invert=1))

        br = np.array(self.transform(res, center, scale, res, invert=1))

        old_x = max(0, ul[0]), min(len(img[0]), br[0])
        old_y = max(0, ul[1]), min(len(img), br[1])
        print("_________")
        print([old_y[0],old_y[1], old_x[0],old_x[1]])
        new_img = img[old_y[0]:old_y[1], old_x[0]:old_x[1],:]

        return scipy.misc.imresize(new_img, res)

    def getFeature(self, box):
        x1,y1,x2,y2 = box
        center = np.array(((x1 + x2) * 0.5, (y1 + y2) * 0.5))
        scale = y2 - y1, x2 - x1
        return center, scale

    def recoverFromHm(self, hm,center,scale):
        res = []
        for nbatch in range(hm.shape[0]):
            tmp_lst = []
            for i in range(hm.shape[-1]):
                index = np.unravel_index(hm[nbatch, :, :, i].argmax(), self.res)
                tmp_lst.append(index[::-1])
            res.append(self.transformPreds(np.stack(tmp_lst), center[nbatch], scale[nbatch], self.res, reverse=1))
        return np.stack(res)

    def _makeGaussian(self, height, width, sigma=3, center=None):
        """ Make a square gaussian kernel.
        size is the length of a side of the square
        sigma is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        if center is None:
            x0 = width // 2
            y0 = height // 2
        else:
            x0 = center[0]
            y0 = center[1]
        return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)

    def generateHeatMap(self, center, scale ,height, width, joints, maxlenght, weight):
        """ Generate a full Heap Map for every joints in an array
        Args:
            height			: Wanted Height for the Heat Map
            width			: Wanted Width for the Heat Map
            joints			: Array of Joints
            maxlenght		: Lenght of the Bounding Box
        """
        joints = self.transformPreds(joints, center=center,scale=scale, res=[height,width])
        num_joints = joints.shape[0]
        hm = np.zeros((height, width, num_joints), dtype=np.float32)
        for i in range(num_joints):
            if not (np.array_equal(joints[i], [-1, -1])) and weight[i] == 1:
                s = int(np.sqrt(maxlenght) * maxlenght * 10 / 4096) + 2
                hm[:, :, i] = self._makeGaussian(height, width, sigma=s, center=(joints[i, 0], joints[i, 1]))
            else:
                hm[:, :, i] = np.zeros((height, width))
        return hm


    @property
    def provide_data(self):
        return [('data',(self.batch_size,3, self.img_size ,self.img_size))]

    @property
    def provide_label(self):
        return [('label',(self.batch_size, self.hm_size,self.hm_size,self.partnum))]

    def get_batch_size(self):

        return self.batch_size
    def reset(self):
        self.cursor = -self.batch_size
        self._randomize()

    def iter_next(self):

        self.cursor += self.batch_size
        return self.cursor < self.num_data

    def _getdata(self):

        assert (self.cursor < self.num_data), "DataIter needs reset."

        data = np.zeros((self.batch_size, 3, self.img_size, self.img_size))
        label = np.zeros((self.batch_size, self.outsize ,self.outsize,self.partnum))

        if self.cursor + self.batch_size <= self.num_data:
            for i in range(self.batch_size):
                name = self.dataset[self.cursor + i]
                joints = self.data_dict[name]['joints']
                box = self.data_dict[name]['box']
                center, scale = self.getFeature(box)
                weight = np.asarray(self.data_dict[name]['weights'])
                img = self.open_img(name)

                crop_img = self.crop(img, center, scale, [self.img_size,self.img_size])

                hm = self.generateHeatMap(center, scale, 64, 64, joints, 64, weight)
                ##still need augment
                if self.normalize:
                    data[i] = np.transpose(crop_img.astype(np.float32) / 255,[2,0,1])
                else:
                    data[i] = np.transpose(crop_img.astype(np.float32) ,[2,0,1])
                label[i] = hm
        else:
            for i in range(self.num_data - self.cursor):

                name = self.dataset[self.cursor + i]
                joints = self.data_dict[name]['joints']
                box = self.data_dict[name]['box']
                center, scale = self.getFeature(box)
                weight = np.asarray(self.data_dict[name]['weights'])
                img = self.open_img(name)

                crop_img = self.crop(img, center, scale, [self.img_size,self.img_size])

                hm = self.generateHeatMap(center, scale, 64, 64, joints, 64, weight)
                ##still need augment
                if self.normalize:
                    data[i] = crop_img.astype(np.float32) / 255
                else:
                    data[i] = crop_img.astype(np.float32)
                label[i] = hm
            pad = self.batch_size - self.num_data + self.cursor
            for i in range(pad):
                name = self.dataset[ i]
                joints = self.data_dict[name]['joints']
                box = self.data_dict[name]['box']
                center, scale = self.getFeature(box)
                weight = np.asarray(self.data_dict[name]['weights'])
                img = self.open_img(name)

                crop_img = self.crop(img, center, scale, [self.img_size,self.img_size])

                hm = self.generateHeatMap(center, scale, 64, 64, joints, 64, weight)
                ##still need augment
                if self.normalize:
                    data[i + self.num_data - self.cursor] = crop_img.astype(np.float32) / 255
                else:
                    data[i + self.num_data - self.cursor] = crop_img.astype(np.float32)
                label[i + self.num_data - self.cursor] = hm


        return mx.nd.array(data), mx.nd.array(label)


    def next(self):

        """return one dict which contains "data" and "label" """

        if self.iter_next():
            data, label = self._getdata()
            data = [data]
            label = [label]

            return DataBatch(data=data, label=label,

                provide_data = self.provide_data,

                provide_label = self.provide_label)
        else:

            raise StopIteration