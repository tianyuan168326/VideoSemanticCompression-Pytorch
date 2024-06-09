action_names = [
    "bend/bow (at the waist)",
    "lie/sleep",
    "run/jog",
    "sit",
    'stand',
    'walk',
    'carry/hold (an object)',
    'cut',
    'drive'
    'put down',
    'ride (e.g., a bike, a car, a horse)',
    'shoot',
    'watch (e.g., TV)',
    'listen to (a person)',
    'talk to (e.g., self, a person, a group)',
    'watch (a person)'
]

def is_action_trigger(result):
    state = False
    for a in action_names:
        if a in result:
            state = True
    return state

import os
def print_my_ava_score(src_file):
    s_list = []
    codec_b = -1
    meta_b = -1
    with open(src_file,'r') as f:
        result_lines = f.readlines()
        for r in result_lines:
            if "codec_b" in r and codec_b == -1:
                codec_b = float(r.split("\t")[-1])
            if "meta_b" in r and meta_b == -1:
                meta_b = float(r.split("\t")[-1])
            if not "PerformanceByCategory/AP@0.5IOU" in r:
                continue
            r_tok = r.split("\t")
            name,score = r_tok[0], r_tok[-1]
            if score == "nan":
                continue
            score = float(score)
            if is_action_trigger(name):
                # print(r)
                s_list +=[score]
        print( os.path.basename(src_file), codec_b + meta_b , sum(s_list)/len(s_list))

src_files = [
    '/data_video/code/lbvu/mmaction/slowfast_h264_35.log',
    '/data_video/code/lbvu/mmaction/slowfast_h264_39.log',
    '/data_video/code/lbvu/mmaction/slowfast_h264_43.log',
    '/data_video/code/lbvu/mmaction/slowfast_h264_47.log',
    '/data_video/code/lbvu/mmaction/slowfast_uvc39.log',
    '/data_video/code/lbvu/mmaction/slowfast_uvc43.log',
    '/data_video/code/lbvu/mmaction/slowfast_uvc47.log',
    '/data_video/code/lbvu/mmaction/slowfast_uvc51.log',
    "/data_video/code/lbvu/mmaction/arcn_h264_35.log",
    "/data_video/code/lbvu/mmaction/arcn_h264_39.log",
    "/data_video/code/lbvu/mmaction/arcn_h264_43.log",
    "/data_video/code/lbvu/mmaction/arcn_h264_47.log",
    "/data_video/code/lbvu/mmaction/arcn_uvc39.log",
    "/data_video/code/lbvu/mmaction/arcn_uvc43.log",
    "/data_video/code/lbvu/mmaction/arcn_uvc47.log",
    "/data_video/code/lbvu/mmaction/arcn_uvc51.log"
]

for s in src_files:
    print_my_ava_score(s)