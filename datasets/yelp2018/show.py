import pickle
import torch as t
import torch_sparse as ts 
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
import numpy as np



file1 = '/home/guoxchen/unlearn/datasets/yelp2018/trn_mat.pkl'
file2 = '/home/guoxchen/unlearn/datasets/yelp2018/adv_mat.pkl'
file3 = '/home/guoxchen/unlearn/datasets/yelp2018/adv_lightgcn_mat.pkl'


def load_one_file( filename, adversarial_attack=False,test_file=False,non_binary=False):
    with open(filename, 'rb') as fs:
        tem = pickle.load(fs)
        if adversarial_attack and (not test_file):
            adv_edges = tem[1] 
            tem = tem[0]                           
        ret = tem if non_binary else (tem != 0).astype(np.float32)
    if type(ret) != coo_matrix:
        ret = sp.coo_matrix(ret)
    ret = ts.SparseTensor.from_scipy(ret)
    return ret



trnMat = load_one_file(file1)
advMat = load_one_file(file2, adversarial_attack=True)
adv_lightgcn_mat = load_one_file(file3, adversarial_attack=True)

print("#################trnMat####################")
print(trnMat)

print("#################advMat####################")
print(advMat)

print("#################adv_lightgcn_mat####################")
print(adv_lightgcn_mat)


