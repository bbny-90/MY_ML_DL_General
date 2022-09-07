import numpy as np
import pandas as pd
import torch


def get_best_teacher(dir_teachers, common_name, teacher_ids, criterion:list):
    min_val = np.inf
    for i in teacher_ids:
        df = pd.read_csv(dir_teachers + common_name + "{}.csv".format(i))
        temp = df[criterion].values
        temp = temp.sum(axis=1)
        m_id = np.argmin(temp)
        m_val = temp[m_id]
        if m_val < min_val:
            min_val = m_val
            min_id = i
    return min_id, min_val

def creat_random_map(j, n):
    """
        https://arxiv.org/pdf/1511.05641.pdf
        see page 4
    """
    assert n >= j
    out = np.arange(n)
    if j<n:
        out[j:] = np.random.choice(j, n - j)
    temp_uniq, temp_cnts = np.unique(out, return_counts=True)
    cnt_dic = {temp_uniq[i]:temp_cnts[i] for i in range(len(temp_uniq))}
    # temp_cnts2 = [temp_dic[i] for i in out]
    return out, cnt_dic

def init_wlayer(new_l, old_l, l1, l2, g1, g2, cnt_dic1, noise_level):
    """
        https://arxiv.org/pdf/1511.05641.pdf
        see page 5
    """
    # # temp = old_l.weight.data.detach().numpy()
    # # # mean = np.mean(temp, axis=0)
    # # std = np.std(temp, axis=0)
    # # nois = np.zeros_like(temp)
    # # num_row, num_col = temp.shape
    # # for i in range(num_col):
    # #     nois[:, i] = np.random.randn(num_row)*std[i]*0.5
    dev = new_l.weight.data.device

    for k in range(l1):
        for j in range(l2):
            # new_l.weight.data[j, k] = (old_l.weight.data[g2[j], g1[k]] + nois[g2[j], g1[k]]) / cnt_dic1[g1[k]]
            new_l.weight.data[j, k] = old_l.weight.data[g2[j], g1[k]] / cnt_dic1[g1[k]]
    temp = new_l.weight.data.detach().numpy()
    # mean = np.mean(temp, axis=0)
    std = np.std(temp, axis=0)
    nois = np.zeros_like(temp)
    num_row, num_col = temp.shape
    for i in range(num_col):
        nois[:, i] = np.random.randn(num_row)*std[i]*noise_level
    temp = torch.from_numpy(nois).float().to(dev)
    new_l.weight.data = new_l.weight.data + temp
    
    new_l.bias.data = old_l.bias.data[g2]
    std = np.std(new_l.bias.data.detach().numpy())
    temp = np.random.randn(new_l.bias.data.shape[0])*std*noise_level
    temp = torch.from_numpy(temp).float().to(dev)
    new_l.bias.data = new_l.bias.data + temp
    # exit()

def creat_net_from_teacher(studnet_layers:list, teacher_nn_layers, noise_level):
    """
        https://arxiv.org/pdf/1511.05641.pdf
    """
    teacher_layers = [i.in_features for i in teacher_nn_layers]
    teacher_layers += [teacher_nn_layers[-1].out_features]
    
    assert studnet_layers[0] == teacher_layers[0]
    assert studnet_layers[-1] == teacher_layers[-1]
    assert len(studnet_layers) == len(teacher_layers)
    num_layers = len(studnet_layers)
    # j = teacher_nn_layers[0].out_features
    # n = studnet_layers[1]
    # assert n >= j
    
    g_all, cnt_dic_all = [], []
    for i in range(num_layers):
        j, n = teacher_layers[i], studnet_layers[i]
        assert n >= j
        gi, cnt_dici = creat_random_map(j, n)
        g_all.append(gi)
        cnt_dic_all.append(cnt_dici)
    
    student_mlp = torch.nn.ModuleList()
    for i in range(len(studnet_layers)-1):
        lini = torch.nn.Linear(studnet_layers[i], studnet_layers[i+1])
        init_wlayer(lini, teacher_nn_layers[i], studnet_layers[i],
                    studnet_layers[i+1], g_all[i], g_all[i+1],
                    cnt_dic_all[i], noise_level)
        student_mlp.append(lini)
    return student_mlp