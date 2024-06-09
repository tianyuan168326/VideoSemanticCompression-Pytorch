import os
import torch
import torchvision
import torchvision.datasets as datasets


def return_somethingv1(ROOT_DATASET):
    filename_categories = 'somethingv1/category.txt'
    root_data = "/data_video/sthv1_frames/sth-sth-v1/20bn-something-something-v1"
    # root_data = "/data_video/sthv1_frames/sth-sth-v1/20bn-something-something-v1"
    filename_imglist_train = 'somethingv1/train.txt'
    filename_imglist_val = 'somethingv1/valid.txt'
    prefix = '{:05d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
def return_somethingv1_q43(ROOT_DATASET):
    filename_categories = 'somethingv1/category.txt'
    root_data = "/media/ps/SSD/tianyuan/sthv1_h265_q43"
    # root_data = "/data_video/sthv1_frames_h265_43"
    
    # root_data = "/data_video/sthv1_frames/sth-sth-v1/20bn-something-something-v1"
    filename_imglist_train = 'somethingv1/train.txt'
    filename_imglist_val = 'somethingv1/valid.txt'
    prefix = '{:05d}.png'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
def return_somethingv1_q35(ROOT_DATASET):
    filename_categories = 'somethingv1/category.txt'
    root_data = "/media/ps/SSD/tianyuan/sthv1_h265_q35"
    # root_data = "/data_video/sthv1_frames_h265_35"
    
    # root_data = "/data_video/sthv1_frames/sth-sth-v1/20bn-something-something-v1"
    filename_imglist_train = 'somethingv1/train.txt'
    filename_imglist_val = 'somethingv1/valid.txt'
    prefix = '{:05d}.png'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
def return_somethingv1_q43_10k(ROOT_DATASET):
    filename_categories = 'somethingv1_10k/category.txt'
    root_data = "/media/ps/SSD/tianyuan/sthv1_h265_q43"
    # root_data = "/data_video/sthv1_frames_h265_43"
    
    # root_data = "/data_video/sthv1_frames/sth-sth-v1/20bn-something-something-v1"
    filename_imglist_train = 'somethingv1_10k/train_10k_used.txt'
    filename_imglist_val = 'somethingv1_10k/val_1k_used.txt'
    prefix = '{:05d}.png'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
def return_ucf101(ROOT_DATASET):
    filename_categories = 'VideoDatasetList/ucf101/category.txt'
    root_data = "/data_video/ucf101_jpg"
    # root_data = "/data_video/sthv1_frames/sth-sth-v1/20bn-something-something-v1"
    filename_imglist_train = 'VideoDatasetList/ucf101/train_tsn_01.txt'
    filename_imglist_val = 'VideoDatasetList/ucf101/test_tsn_01.txt'
    prefix = '{:06d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_ucf101_crf39(ROOT_DATASET):
    filename_categories = 'ucf101/category.txt'
    root_data = "/media/ps/SSD/tianyuan/ucf101_jpg_crf39"
    # root_data = "/data_video/sthv1_frames/sth-sth-v1/20bn-something-something-v1"
    filename_imglist_train = 'ucf101/train_tsn_01.txt'
    filename_imglist_val = 'ucf101/test_tsn_01.txt'
    prefix = '{:06d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
def return_ucf101_crf43(ROOT_DATASET):
    filename_categories = 'ucf101/category.txt'
    root_data = "/media/ps/SSD/tianyuan/ucf101_jpg_crf43"
    # root_data = "/data_video/sthv1_frames/sth-sth-v1/20bn-something-something-v1"
    filename_imglist_train = 'ucf101/train_tsn_01.txt'
    filename_imglist_val = 'ucf101/test_tsn_01.txt'
    prefix = '{:06d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
def return_ucf101_crf47(ROOT_DATASET):
    filename_categories = 'ucf101/category.txt'
    root_data = "/data_video/ucf101_jpg_crf47/ucf101_jpg"
    # root_data = "/data_video/sthv1_frames/sth-sth-v1/20bn-something-something-v1"
    filename_imglist_train = 'ucf101/train_tsn_01.txt'
    filename_imglist_val = 'ucf101/test_tsn_01.txt'
    prefix = '{:06d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
def return_ucf101_crf51(ROOT_DATASET):
    filename_categories = 'ucf101/category.txt'
    root_data = "/media/ps/SSD/tianyuan/ucf101_jpg_crf51"
    # root_data = "/data_video/sthv1_frames/sth-sth-v1/20bn-something-something-v1"
    filename_imglist_train = 'ucf101/train_tsn_01.txt'
    filename_imglist_val = 'ucf101/test_tsn_01.txt'
    prefix = '{:06d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
def return_ucf101_crf27(ROOT_DATASET):
    filename_categories = 'ucf101/category.txt'
    root_data = "/media/ps/SSD/tianyuan/ucf101_jpg_crf27"
    # root_data = "/data_video/sthv1_frames/sth-sth-v1/20bn-something-something-v1"
    filename_imglist_train = 'ucf101/train_tsn_01.txt'
    filename_imglist_val = 'ucf101/test_tsn_01.txt'
    prefix = '{:06d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
def return_ucf101_br5(ROOT_DATASET):
    filename_categories = 'VideoDatasetList/ucf101/category.txt'
    root_data = "/data_video/ucf101_jpg_br5"
    # root_data = "/data_video/sthv1_frames/sth-sth-v1/20bn-something-something-v1"
    filename_imglist_train = 'VideoDatasetList/ucf101/train_tsn_01.txt'
    filename_imglist_val = 'VideoDatasetList/ucf101/test_tsn_01.txt'
    prefix = '{:06d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_ucf101_br10(ROOT_DATASET):
    filename_categories = 'ucf101/category.txt'
    root_data = "/media/ps/SSD/tianyuan/ucf101_jpg_br10"
    # root_data = "/data_video/sthv1_frames/sth-sth-v1/20bn-something-something-v1"
    filename_imglist_train = 'ucf101/train_tsn_01.txt'
    filename_imglist_val = 'ucf101/test_tsn_01.txt'
    prefix = '{:06d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_ucf101_br100(ROOT_DATASET):
    filename_categories = 'ucf101/category.txt'
    root_data = "/media/ps/SSD/tianyuan/ucf101_jpg_br100"
    # root_data = "/data_video/sthv1_frames/sth-sth-v1/20bn-something-something-v1"
    filename_imglist_train = 'ucf101/train_tsn_01.txt'
    filename_imglist_val = 'ucf101/test_tsn_01.txt'
    prefix = '{:06d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
def return_ucf101_br20(ROOT_DATASET):
    filename_categories = 'ucf101/category.txt'
    root_data = "/media/ps/SSD/tianyuan/ucf101_jpg_br20"
    # root_data = "/data_video/sthv1_frames/sth-sth-v1/20bn-something-something-v1"
    filename_imglist_train = 'ucf101/train_tsn_01.txt'
    filename_imglist_val = 'ucf101/test_tsn_01.txt'
    prefix = '{:06d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_ucf101_br40(ROOT_DATASET):
    filename_categories = 'ucf101/category.txt'
    root_data = "/media/ps/SSD/tianyuan/ucf101_jpg_br40"
    # root_data = "/data_video/sthv1_frames/sth-sth-v1/20bn-something-something-v1"
    filename_imglist_train = 'ucf101/train_tsn_01.txt'
    filename_imglist_val = 'ucf101/test_tsn_01.txt'
    prefix = '{:06d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix



def return_ucf101_br200(ROOT_DATASET):
    filename_categories = 'ucf101/category.txt'
    root_data = "/media/ps/SSD/tianyuan/ucf101_jpg_br200"
    # root_data = "/data_video/sthv1_frames/sth-sth-v1/20bn-something-something-v1"
    filename_imglist_train = 'ucf101/train_tsn_01.txt'
    filename_imglist_val = 'ucf101/test_tsn_01.txt'
    prefix = '{:06d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_kinetics400(ROOT_DATASET):
    filename_categories = 'kinetics400/category.txt'
    root_data = "/data_video/important_dataset/kinetics-400-jpg"
    filename_imglist_train = 'kinetics400/train.txt'
    filename_imglist_val = 'kinetics400/valid.txt'
    prefix = 'image_{:05d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
def return_kinetics400video(ROOT_DATASET):
    filename_categories = '/data_video/code/lbvu/VideoDatasetList/kinetics20k/category.txt'
    root_data = "/data_video/k400_val"
    filename_imglist_train = '/data_video/code/lbvu/data_process/k400_ds/kinetics400_mmaction_video.txt'
    filename_imglist_val = '/data_video/code/lbvu/data_process/k400_ds/kinetics400_mmaction_video.txt'
    prefix = 'image_{:05d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
def return_kinetics200(ROOT_DATASET):
    filename_categories = 'kinetics200/category.txt'
    root_data = "/data_video/important_dataset/kinetics-400-jpg"
    filename_imglist_train = 'kinetics200/train_k200.txt'
    filename_imglist_val = 'kinetics200/test_k200.txt'
    prefix = 'image_{:05d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
def return_k20k_br5(ROOT_DATASET):
    filename_categories = 'kinetics20k/category.txt'
    root_data = "/media/ps/SSD/tianyuan/k20k_br5"
    filename_imglist_train = 'kinetics20k/train.txt'
    filename_imglist_val = 'kinetics20k/validation.txt'
    prefix = 'image_{:05d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_k60k_brrand(ROOT_DATASET):
    filename_categories = 'VideoDatasetList/category.txt'
    root_data = ""
    filename_imglist_train = 'VideoDatasetList/train_hq.txt'
    filename_imglist_val = '=='
    prefix = 'image_{:05d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_k60k_brrand_a6000(ROOT_DATASET):
    filename_categories = '/home/ubuntu/research/A6000_lbvu/VideoDatasetList/kinetics20k/category.txt'
    root_data = ""
    filename_imglist_train = '/home/ubuntu/research/A6000_lbvu/VideoDatasetList/kinetics50k/train_hq_A6000.txt'
    filename_imglist_val = '=='
    prefix = 'image_{:05d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
def return_k60k_brrand_cloud(ROOT_DATASET):
    filename_categories = '/root/autodl-tmp/video_semantic_coding/VideoDatasetList/kinetics20k/category.txt'
    root_data = ""
    filename_imglist_train = '/root/autodl-tmp/video_semantic_coding/VideoDatasetList/kinetics50k/train_hq_cloud.txt'
    filename_imglist_val = '=='
    prefix = 'image_{:05d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_diving48(ROOT_DATASET):
    filename_categories = 'diving48/category.txt'
    root_data = "/data_video/diving48/frames"
    filename_imglist_train = 'diving48/train_videofolder.txt'
    filename_imglist_val = 'diving48/val_videofolder.txt'
    prefix = 'frames{:05d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
def return_vimeo90k(ROOT_DATASET):
    filename_categories = '/data_video/code/lbvu/VideoDatasetList/Vimeo90K/category.txt'
    root_data = ""
    filename_imglist_train = '/data_video/code/lbvu/VideoDatasetList/Vimeo90K/train.txt'
    filename_imglist_val = '=='
    prefix = 'im{:d}.png'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
def return_somethingv1test(ROOT_DATASET):
    filename_categories = 'somethingv1_test/category.txt'
    root_data = "/data_video/sthv1_frames/sth-sth-v1/20bn-something-something-v1"
    filename_imglist_train = 'somethingv1_test/train.txt'
    filename_imglist_val = 'somethingv1_test/valid.txt'
    prefix = '{:05d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
def return_somethingv2(ROOT_DATASET):
    filename_categories = ROOT_DATASET + 'somethingv2/category.txt'
    # root_data = "/data_video/important_dataset/sthv2-frames"
    root_data = "/media/ps/SSD/tianyuan/sthv2-frames"
    filename_imglist_train = 'somethingv2/train.txt'
    filename_imglist_val = 'somethingv2/valid.txt'
    prefix = '{:06d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset,ROOT_DATASET):
    dict_single = { 
        'somethingv1':return_somethingv1,
        'somethingv1_q43':return_somethingv1_q43,
        'somethingv1_q35':return_somethingv1_q35,
        'somethingv1_q43_10k':return_somethingv1_q43_10k,
        'somethingv1_test':return_somethingv1test,
        'somethingv2':return_somethingv2,
        'k400':return_kinetics400,
        'k400_video':return_kinetics400video,
        "k200":return_kinetics200,
        "diving48":return_diving48,
        "vimeo90k":return_vimeo90k,
        "ucf101":return_ucf101,
        "ucf101_crf39":return_ucf101_crf39,
        "ucf101_crf43":return_ucf101_crf43,
        "ucf101_crf47":return_ucf101_crf47,
        "ucf101_crf51":return_ucf101_crf51,
        "ucf101_crf27":return_ucf101_crf27,
        "ucf101_br5":return_ucf101_br5,
        "ucf101_br10":return_ucf101_br10,
        "ucf101_br20":return_ucf101_br20,
        "ucf101_br40":return_ucf101_br40,
        "ucf101_br100":return_ucf101_br100,
        "ucf101_br200":return_ucf101_br200,
        "ucf101_brrand":return_ucf101_crf51,
        "k20k_brrand":return_k20k_br5,
        "k60k_brrand_a6000":return_k60k_brrand_a6000,
        "k60k_brrand_cloud":return_k60k_brrand_cloud,
        "k60k_brrand":return_k60k_brrand,
        

     }
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](ROOT_DATASET)
    else:
        print('Unknown dataset '+dataset,"set dataset info to dummy value, None")
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = "", "","","",""

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    file_categories = os.path.join(ROOT_DATASET, file_categories)
    
    try:
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    except:
        categories  = [str(i) for i in range(101)]
    return categories, file_imglist_train, file_imglist_val, root_data, prefix

