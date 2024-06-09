import torch.utils.data as data
import torch

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import random
import skvideo.io
import numpy as np
import os
import random
import time
import torchvision
import copy
class VideoRecord(object):
    def __init__(self, row):
        self._data = row
        

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])
    
    def get_index_list(self,begin_index = 1):
        if self.is_last:
            if self.len <self.seg_len:
                offsets = list(range(self.begin_i, self.begin_i + self.len))
                for pad_i in range(self.seg_len - self.len):
                    offsets += [self.begin_i + self.len-1]
                return np.array(offsets) + begin_index
        else:
            offsets = list(range(self.begin_i, self.begin_i + self.len))
            return np.array(offsets) + begin_index
    def set_first_index(self,begin_i,len,seg_len,is_last):
        self.begin_i = begin_i
        self.len = len
        self.seg_len = seg_len
        self.is_last = is_last
file_random_name = str(time.time())
keyint = 8
q = 61


class VideoDataSetSegOnline(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments, image_tmpl, new_length = 1, transform=None, 
                 random_shift=True, test_mode=False,dataset="",crf =22,
                 multi_clip_test = False,
                 dense_sample=False,
                 only_clean = False,
                 only_first_half = False,
                 dup_num = 1,
                 begin_index = 1,
                 num_clips=1,number_id = None,return_video_len=True):
        self.begin_index = begin_index
        self.only_clean = only_clean ### True ||| duplicate compressed as clean
        self.only_first_half = only_first_half
        self.dup_num = dup_num
        self.root_path = root_path
        self.list_file = list_file
        self.new_length = new_length
        self.num_segments = num_segments
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.multi_clip_test = multi_clip_test
        self.dataset = dataset
        self.crf = crf
        ### for our VTM enhanced algorithm, we just create the dataset with crf in [23,32]
        if self.crf>32:
            ## for FVC, crf>32, we give dummy compressed video data
            self.crf = 32
        self.num_clips = num_clips
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.return_video_len = return_video_len ## is return the length of the oiginal video, used for estimating the bitrate
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        self.number_id = number_id
        self._parse_list()

    def _load_image(self, directory, idx):
        try:
            img_path = os.path.join(self.root_path, directory, self.image_tmpl.format(idx))
            compressed_img = [Image.open(img_path).convert('RGB')]
        except Exception:
            print(('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx))))
            img_path = os.path.join(self.root_path, directory, self.image_tmpl.format(1))
            compressed_img =  [Image.open(img_path).convert('RGB')]
        return compressed_img,img_path
    def _load_clean_image(self, directory, idx):
        
        clean_tmpl = self.image_tmpl
        try:
            clean_path = os.path.join( self.root_path,directory, clean_tmpl.format(idx))
            clean_img = [Image.open(clean_path).convert('RGB')]
        except Exception:
            print('exception because error clean loading image:', clean_path)

        return clean_img,""

    def _parse_list(self):
        # check the frame number is large >3:
        # usualy it is [video_id, num_frames, class_idx]
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[1])>=3]
        if self.number_id:
            tmp = [item for item in tmp if str(self.number_id) in str(item[0]) ]
        # tmp = [item for item in tmp if 130 == int(item[1]) ]
        self.video_list = [VideoRecord(item) for item in tmp]
        Seg = self.num_segments
        new_list = []
        for v in self.video_list:
            num_f = v.num_frames
            if self.only_first_half:
                num_f = num_f//2
            seg_num = num_f//Seg
            seg_last = num_f%Seg
            for i in range(seg_num):
                new_v = copy.deepcopy(v)
                ##begin_i,len,seg_len,is_last
                new_v.set_first_index(i* Seg,Seg,Seg,False)
                new_list += [new_v]
            if seg_last >0:
                new_v = copy.deepcopy(v)
                new_v.set_first_index(seg_num* Seg,seg_last,Seg,True)
                new_list += [new_v]

        self.video_list = []
        for i in range (self.dup_num):
            self.video_list.extend(new_list)
        print(('video number:%d'%(len(self.video_list))))

  
    def _get_val_indices(self, record):
        """Sampling for validation set
        Sample the middle frame from each video segment
        """
        ### return all frames
        first_i = record.get_index_list(begin_index = self.begin_index)
        # print("frames index:",first_i)
        # exit()
        return first_i
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        ###TSN style
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = list(range(record.num_frames))
            # offsets = np.array(offsets)
            offsets_padding = np.zeros((self.num_segments - record.num_frames,)).tolist()
            offsets = offsets_padding+offsets
            offsets = np.array(offsets)
        return offsets + 1
    
    def _get_k400_train_indices(self, record):
        interval = 8
        sample_pos = max(1, 1 + record.num_frames - 64)
        t_stride = 64 // self.num_segments
        start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
        offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
        # print(offsets)
        return np.array(offsets) + 1


    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder
        while not os.path.exists(os.path.join(self.root_path, record.path, self.image_tmpl.format(1))):
            print("not exist",(os.path.join(self.root_path, record.path, self.image_tmpl.format(1))))
            # exit()
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
        
        segment_indices =  self._get_val_indices(record)
        
        
        x,y,z,video_len =  self.get(record, segment_indices)
        # print("x",x.size(),x.device,torch.utils.data.get_worker_info().id)
        if self.return_video_len:
            return x,y,z,video_len
        else:
            return x,y,z

    def get(self, record, indices):
        batch_crf = self.crf
        if batch_crf == -1:
            batch_crf = random.sample([32,29,26,23],1)[0]
        if self.num_clips > 1:
            raise Exception("...num clip >1")
            process_data_final_compressed = []
            process_data_final_clean = []
            for k in range(self.num_clips):
                images_compressed = list()
                images_clean = list()
                for seg_ind in indices[k]:
                    p = int(seg_ind)
                    for i in range(self.new_length):
                        seg_imgs_clean = self._load_clean_image(record.path, p)
                        images_clean.extend(seg_imgs_clean)
                        if p < record.num_frames:
                            p += 1
                    p_pre = p-1
                    if p_pre <1:
                        p_pre = 1
                    p_next = p+1
                    if p_next > record.num_frames:
                        p_next = record.num_frames
                    
                    seg_imgs_compressed= self._load_image(record.path, p_pre)
                    images_compressed.extend(seg_imgs_compressed)
                    seg_imgs_compressed= self._load_image(record.path, p)
                    images_compressed.extend(seg_imgs_compressed)
                    seg_imgs_compressed= self._load_image(record.path, p_next)
                    images_compressed.extend(seg_imgs_compressed)
                process_data, label = self.transform((images_compressed, record.label))
                process_data_final_compressed.append(process_data)
                process_data, label = self.transform((seg_imgs_clean, record.label))
                process_data_final_clean.append(process_data)
            process_data_final_compressed = torch.stack(process_data_final_compressed, 0)#
            process_data_final_clean = torch.stack(process_data_final_clean, 0)#
            return process_data_final_compressed,process_data_final_clean, label
        elif self.only_clean:
            images_compressed = list()
            img_path_l = []
            for seg_ind in indices:
                p = int(seg_ind)
                for i in range(self.new_length):
                    tmp,img_path = self._load_image(record.path, p)
                    images_compressed.extend(tmp)
                    img_path_l += [img_path]
                    if p < record.num_frames:
                        p += 1
                
            
            images = images_compressed
            process_data, label = self.transform((images, record.label))

            h,w = process_data.size(-2), process_data.size(-1)
            clip_len =  len(indices )

            images = process_data.reshape(clip_len,3,h,w)
            process_data_com = images.reshape(-1,h,w)
            video_len = record.num_frames
            return process_data_com,process_data_com, label,img_path_l
        else:
            images_compressed = list()
            images_clean = list()
            img_path_l = []

            for seg_ind in indices:
                p = int(seg_ind)
                for i in range(self.new_length):
                    if self.dataset == 'hevc_bcde':
                        path_clean = record.path
                        vid_name = path_clean.split("/")[-2] # extract the second-to-last element from the path
                        clip_index = path_clean.split("/")[-1] # extract the last element from the path
                        path_comp =  "/data_extend_yuan/HEVC_test_sequence_VTM_results_yuv444/{}_qp{}/{}".format(vid_name,batch_crf, clip_index)
                        tmp,img_path = self._load_image(path_comp, p)
                        images_compressed.extend(tmp)
                        img_path_l += [img_path]
                        tmp,_= self._load_image(path_clean, p)
                        images_clean.extend(tmp)
                    elif self.dataset == 'hevc_bcde_HM':
                        path_clean = record.path
                        vid_name = path_clean.split("/")[-2] # extract the second-to-last element from the path
                        clip_index = path_clean.split("/")[-1] # extract the last element from the path
                        path_comp =  "/data_extend_yuan/HEVC_test_sequence_HM_results_yuv444/{}_qp{}/{}".format(vid_name,batch_crf, clip_index)
                        tmp,img_path = self._load_image(path_comp, p)
                        images_compressed.extend(tmp)
                        img_path_l += [img_path]
                        tmp,_= self._load_image(path_clean, p)
                        images_clean.extend(tmp)
                    elif self.dataset == 'hevc_bcde_JM':
                        path_clean = record.path
                        vid_name = path_clean.split("/")[-2] # extract the second-to-last element from the path
                        clip_index = path_clean.split("/")[-1] # extract the last element from the path
                        path_comp =  "/data_extend_yuan/HEVC_test_sequence_JM_results_yuv444/{}_qp{}/{}".format(vid_name,batch_crf, clip_index)
                        tmp,img_path = self._load_image(path_comp, p)
                        images_compressed.extend(tmp)
                        img_path_l += [img_path]
                        tmp,_= self._load_image(path_clean, p)
                        images_clean.extend(tmp)
                    elif self.dataset == 'REDS':
                        path_clean = record.path
                        path_comp = path_clean.replace("REDS_train_sharp/REDS_train_sharp/train_sharp","REDS_train_sharp/REDS_train_sharp/train_sharp_VTM_qp{}".format(batch_crf))
                        tmp,img_path = self._load_image(path_comp, p)
                        images_compressed.extend(tmp)
                        img_path_l += [img_path]
                        tmp,_= self._load_image(path_clean, p)
                        images_clean.extend(tmp)
                    elif self.dataset == 'MFQE_GOP10':
                        path_clean = record.path
                        path_comp = path_clean.replace("MFQEV2/train_108/decompressed_imgs","MFQEV2/train_108/decompressed_imgs_VTM_qp{}".format(batch_crf))
                        tmp,img_path = self._load_image(path_comp, p)
                        images_compressed.extend(tmp)
                        img_path_l += [img_path]
                        tmp,_= self._load_image(path_clean, p)
                        images_clean.extend(tmp)
                    elif self.dataset == 'REDS_GOP100':
                        path_clean = record.path
                        path_comp = path_clean.replace("REDS_train_sharp/REDS_train_sharp/train_sharp","REDS_train_sharp/REDS_train_sharp/train_sharp_VTM_G100_qp{}".format(batch_crf))
                        tmp,img_path = self._load_image(path_comp, p)
                        images_compressed.extend(tmp)
                        img_path_l += [img_path]
                        tmp,_= self._load_image(path_clean, p)
                        images_clean.extend(tmp)
                    else:
                        tmp,img_path = self._load_image(record.path, p)
                        images_compressed.extend(tmp)
                        img_path_l += [img_path]
                        tmp,_= self._load_clean_image(record.path, p)
                        images_clean.extend(tmp)
                    
                    if p < record.num_frames:
                        p += 1
                
                
            images = images_clean+images_compressed
            process_data, label = self.transform((images, record.label))
            h,w = process_data.size(-2), process_data.size(-1)
            clip_len =  len(indices )
            # print(process_data.size())
            images = process_data.reshape(clip_len+clip_len,3,h,w)
            process_data_clean = images[0:clip_len].reshape(-1,h,w)
            process_data_com = images[clip_len:].reshape(-1,h,w)
            video_len = record.num_frames
            return process_data_com,process_data_clean, label,img_path_l

    def __len__(self):
        return len(self.video_list)
