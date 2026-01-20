import torch as t
import Utils.time_logger as logger
from Utils.time_logger import log
from params import args
# from model import LightGCN
from data_handler import DataHandler
import numpy as np
import pickle
import os
import setproctitle
from scipy.sparse import coo_matrix
import random
import torch_sparse as ts

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.adversarial_attack = False

t.manual_seed(args.seed)
t.cuda.manual_seed_all(args.seed)
t.backends.cudnn.deterministic = True
np.random.seed(args.seed)
random.seed(args.seed)


handler = DataHandler()
handler.load_data(drop_rate=0.0, adv_attack=False)

# save_path = '../../datasets/ml-1m/adv_rd_mat.pkl'

# save_path = '../../datasets/sparse_yelp/adv_lightgcn_mat.pkl'
# save_path = '../../datasets/sparse_gowalla/adv_lightgcn_mat.pkl'
# save_path = '../../datasets/sparse_amazon/adv_lightgcn_mat.pkl'
save_path = '../../datasets/yelp2018/adv_simgcl0.6_mat.pkl'


# model_path = '../outModels/yelp/retrain/retrain_yelp_lightgcn_reg1e-6_lr1e-3_b4096_ep300_dim128_ly3.mod'
# model_path = '../outModels/gowalla/retrain/retrain_gowalla_lightgcn_reg1e-7_lr1e-3_b4096_ep300_dim128_ly3.mod'
# model_path = '../outModels/amazon/retrain/retrain_amazon_lightgcn_reg1e-8_lr1e-3_b4096_ep300_dim128_ly3.mod'
model_path = '../outModels/yelp2018/retrain/retrain_yelp2018_simgcl_reg1e-6_ssl1e-2_esp2e-1_t1e-1_v1_lr1e-3_b4096_ep300_dim128_ly3.mod'



def find_least_related_edges(model, handler, save_path):
    model.is_training = False
    usr_embeds, itm_embeds = model.forward(handler.torch_adj)
    least_related_edges = [[], []]
    for i in range(args.user):
        usr_embed = usr_embeds[i]
        preds = usr_embed @ itm_embeds.T
        j = t.argmin(preds).item()
        least_related_edges[0].append(i)
        least_related_edges[1].append(j)
    
    for j in range(args.item):
        itm_embed = itm_embeds[j]
        preds = itm_embed @ usr_embeds.T
        i = t.argmin(preds).item()
        least_related_edges.append((i, j))
        least_related_edges[0].append(i)
        least_related_edges[1].append(j)
    # rows = handler.trn_mat.row
    # cols = handler.trn_mat.col
    rows = handler.ori_trn_mat.row
    cols = handler.ori_trn_mat.col


    log('Original number of edges %d' % len(rows))
    rows = np.concatenate([rows, least_related_edges[0]])
    cols = np.concatenate([cols, least_related_edges[1]])
    vals = np.ones_like(rows)
    log('New number of edges %d' % len(rows))
    adv_adj = coo_matrix((vals, (rows, cols)), shape=[args.user, args.item])
    with open(save_path, 'wb') as fs:
        pickle.dump((adv_adj, least_related_edges), fs)



def find_least_related_edges_smp(model, handler, save_path, ratio=0.6):
    model.is_training = False
    usr_embeds, itm_embeds = model.forward(handler.torch_adj)
    least_related_edges = [[], []]    
            
    rd_list = np.random.permutation(args.user)
    user_set = rd_list[:int(ratio*args.user)]
    
    rd_list = np.random.permutation(args.item)
    item_set = rd_list[:int(ratio*args.item)]


    for i in user_set:
        usr_embed = usr_embeds[i]
        preds = usr_embed @ itm_embeds.T
        j = t.argmin(preds).item()
        least_related_edges[0].append(i)
        least_related_edges[1].append(j)
    
    for j in item_set:
        itm_embed = itm_embeds[j]
        preds = itm_embed @ usr_embeds.T
        i = t.argmin(preds).item()
        least_related_edges.append((i, j))
        least_related_edges[0].append(i)
        least_related_edges[1].append(j)
    # rows = handler.trn_mat.row
    # cols = handler.trn_mat.col
    rows = handler.ori_trn_mat.row
    cols = handler.ori_trn_mat.col


    log('Original number of edges %d' % len(rows))
    rows = np.concatenate([rows, least_related_edges[0]])
    cols = np.concatenate([cols, least_related_edges[1]])
    vals = np.ones_like(rows)
    log('New number of edges %d' % len(rows))
    adv_adj = coo_matrix((vals, (rows, cols)), shape=[args.user, args.item])
    with open(save_path, 'wb') as fs:
        pickle.dump((adv_adj, least_related_edges), fs)        

def load_model(load_model):
    ckp = t.load(load_model)
    model = ckp['model']
    return model




model = load_model(model_path)
find_least_related_edges_smp(model, handler, save_path, 0.6)
# find_least_related_edges(model, handler, save_path)



