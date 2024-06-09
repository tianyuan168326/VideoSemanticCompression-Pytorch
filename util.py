import torchvision



import time
import torch.nn.functional as F
import torch.nn as nn
import torch
import numbers
import math
def save_batch_img(imgs,p,t= 8):
    ### t c h w
    imgs = torchvision.utils.make_grid(imgs,nrow=t,normalize =True)
    torchvision.utils.save_image(imgs,p)
def save_attention_mask(src_input,vis_probs,path,time_inter = 10000):
    # time_inter = 1
    ## C T H W ## N T H W
    c,t,h,w = src_input.size()
    if  ("0" in str(src_input.device) or "cpu" in str(src_input.device)) and int(time.time())%time_inter == 0:
        src_input_1st = src_input[:,:,:,:].transpose(0,1)
        vis = [src_input_1st]
        if vis_probs:
            for vis_prob_1st in vis_probs:
                # print("vis_prob_1st",vis_prob_1st.size())
                # vis_prob_1st = F.upsample(vis_prob_1st.unsqueeze(0),size=(t,h,w),mode="trilinear")
                # vis_prob_1st = vis_prob_1st.unsqueeze(2).repeat(1,1,3,1,1).reshape(-1,3,h,w)
                #t 3 h w
                if vis_prob_1st.size(0)  ==1:
                    vis_prob_1st = vis_prob_1st.repeat(3,1,1,1)
                vis += [vis_prob_1st.transpose(0,1)]
        vis = torch.cat(vis,dim=0)
        save_batch_img(vis,path,t)


from prettytable import PrettyTable

def count_parameters(model,black_key =[],only_key = ""):
    str = ''
    table = PrettyTable(["Modules", "Parameters","p-percetage"])
    total_params = 0
    for name, parameter in model.named_parameters():
        continue_flag= False
        for key in black_key:
            if key in name or not only_key in name:
                continue_flag =True
        if continue_flag:
            continue
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params+=param
    for name, parameter in model.named_parameters():
        continue_flag= False
        for key in black_key:
            if key in name or not only_key in name:
                continue_flag =True
        if continue_flag:
            continue
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param, '{:.1%}'.format(param/total_params)])
    # print(table)
    str += table.__str__()
    str += '\n'
    str += f"Total Trainable Params: {total_params} \n"
    total_params = 0
    for n, parameter in model.named_parameters():
        continue_flag= False
        if ".f_net." in n\
            :
            if not parameter.requires_grad: continue
            param = parameter.numel()
            total_params+=param
    str += f"Decoder Total Trainable Params: {total_params} \n"
    
    
    total_params = 0
    for n, parameter in model.named_parameters():
        continue_flag= False
        if ".sem_net." in n\
            :
            if not parameter.requires_grad: continue
            param = parameter.numel()
            total_params+=param
    str += f"sem_net Total Trainable Params: {total_params} \n"

    
    return str