import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from params import args
import scipy.sparse as sp
from Utils.time_logger import log
import torch as t
import torch.utils.data as data
import torch_sparse as ts
import random

class DataHandler:
    def __init__(self, adv_type=args.adv_method):
        if args.data == 'ml1m':
            predir = './datasets/ml-1m' + '/'
        elif args.data == 'ml10m':
            predir = './datasets/ml-10m/'
        elif args.data == 'yelp2018':
            predir = './datasets/yelp2018/'

        elif args.data == 'yelp':
            predir = './datasets/sparse_yelp/'      
        elif args.data == 'gowalla':
            predir = './datasets/sparse_gowalla/'                      
        elif args.data == 'amazon':
            predir = './datasets/sparse_amazon/'    

        # self.trn_file = predir + 'trn_mat.pkl'
        if args.adversarial_attack:            
            print("##########using the least adv_mat#############")                
            self.trn_file = predir + f'adv_{adv_type}_mat.pkl'
            
        else:
            self.trn_file = predir + 'trn_mat.pkl'

        self.tst_file = predir + 'tst_mat.pkl'

        t.manual_seed(args.seed)
        t.cuda.manual_seed_all(args.seed)
        t.backends.cudnn.deterministic = True
        np.random.seed(args.seed)
        random.seed(args.seed)

    def _load_one_file(self, filename, test_file=False,non_binary=False):
        print(f"################here _load_one_file##################")
        with open(filename, 'rb') as fs:
            tem = pickle.load(fs)
            if args.adversarial_attack and (not test_file):
                print(f"################here load self.adv_edges##################")
                self.adv_edges = tem[1] 
                tem = tem[0]                           
            ret = tem if non_binary else (tem != 0).astype(np.float32)
        if type(ret) != coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret

    def _normalize_adj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        if mat.shape[0] == mat.shape[1]:
            return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()
        else:
            tem = d_inv_sqrt_mat.dot(mat)
            col_degree = np.array(mat.sum(axis=0))
            d_inv_sqrt = np.reshape(np.power(col_degree, -0.5), [-1])
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
            d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
            return tem.dot(d_inv_sqrt_mat).tocoo()
    
    def _scipy_to_torch_adj(self, mat):
        # make cuda tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).cuda()
    def _scipy_to_torch_sparse_adj(self, mat):
        ret = ts.SparseTensor.from_scipy(mat).cuda()
        return ret
    
    def _make_torch_adj(self, mat, self_loop=False):
        # make ui adj
        a = sp.csr_matrix((args.user, args.user))
        b = sp.csr_matrix((args.item, args.item))
        bi_mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        bi_mat = (bi_mat != 0) * 1.0
        if self_loop:
            bi_mat = (bi_mat + sp.eye(bi_mat.shape[0])) * 1.0
        bi_mat = self._normalize_adj(bi_mat)
        uni_mat = self._normalize_adj(mat)
        return self._scipy_to_torch_adj(uni_mat), self._scipy_to_torch_adj(bi_mat), self._scipy_to_torch_sparse_adj(bi_mat)
    



    def _make_mask(self, drp_users, drp_items):
        mask = t.zeros(args.user + args.item )
        for i in range(len(drp_users)):
            mask[drp_users[i]] = 1.
            mask[drp_items[i]] = 1.
        return mask.reshape([-1,1]).cuda()
    
    def adversarial_edges_drop(self,adv_mat, adv_edges):
        ori_mat = adv_mat.astype(np.int32).tocoo()
        
        drp_rows = np.array(adv_edges[0])
        drp_cols = np.array(adv_edges[1])
        drp_vals = np.ones(len(adv_edges[1]), dtype=int)

        drp_mat = coo_matrix((drp_vals, (drp_rows, drp_cols)), shape=adv_mat.shape)


        pk_mat = (ori_mat - drp_mat).tocoo()

        
        dropped_users = adv_edges[0]
        dropped_items = (drp_cols + args.user).tolist()
        mask = self._make_mask(dropped_users, dropped_items )
        


        return  pk_mat.astype(np.float32), mask, drp_mat.astype(np.float32),  (list(drp_rows), list(drp_cols)), ( list(pk_mat.row),  list(pk_mat.col)  )


    
    def random_drop_edges(self, trn_mat, rate, adv_attack=False):
        # t.manual_seed(args.seed)
        # t.cuda.manual_seed_all(args.seed)
        # t.backends.cudnn.deterministic = True
        # np.random.seed(args.seed)
        # random.seed(args.seed)

        rows = trn_mat.row
        cols = trn_mat.col
        vals = trn_mat.data
        length = rows.shape[0]
        
        
        rd_list = np.random.permutation(length)
        picked_edges = rd_list[:int((1-rate)*length)]
        droped_edges = rd_list[int((1-rate)*length):]

        pk_rows = rows[picked_edges]
        pk_cols = cols[picked_edges]
        pk_vals = vals[picked_edges]

        drp_rows = rows[droped_edges]
        drp_cols = cols[droped_edges]
        drp_vals = vals[droped_edges]
        

        dropped_users = t.tensor(drp_rows).tolist()
        dropped_items = (t.tensor(drp_cols) + args.user).tolist()

        mask = self._make_mask(dropped_users, dropped_items )

        pk_mat = coo_matrix((pk_vals, (pk_rows, pk_cols)), shape=trn_mat.shape)
        drp_mat = coo_matrix((drp_vals, (drp_rows, drp_cols)), shape=trn_mat.shape)


        return pk_mat, mask, drp_mat,  (list(drp_rows), list(drp_cols)), ( list(pk_rows),  list(pk_cols)  )

    def load_data(self, drop_rate=0.0, adv_attack=False):
        ori_trn_mat = self._load_one_file(self.trn_file)
        tst_mat = self._load_one_file(self.tst_file, test_file=True)

        self.edges_num = ori_trn_mat.row.shape[0]
        args.user, args.item = ori_trn_mat.shape

        self.ori_trn_mat = ori_trn_mat

        _ , self.torch_adj , self.ts_ori_adj = self._make_torch_adj(ori_trn_mat)

        if drop_rate > 0:            
            if adv_attack:
                pk_trn_mat, self.mask,  self.drp_mat, self.dropped_edges, self.picked_edges = self.adversarial_edges_drop(ori_trn_mat, self.adv_edges)
            else:
                pk_trn_mat, self.mask,  self.drp_mat, self.dropped_edges, self.picked_edges = self.random_drop_edges(ori_trn_mat ,drop_rate, False)

            
            self.torch_uni_adj, self.torch_adj, self.ts_pk_adj = self._make_torch_adj(pk_trn_mat)
            _, _, self.ts_drp_adj = self._make_torch_adj(self.drp_mat)

            trn_mat = pk_trn_mat   
            print("##############here in drop_rate >0#################")
        else:
            print("##############here in drop_rate <=0#################")
            if adv_attack:            
                pk_trn_mat, self.mask,  self.drp_mat, self.dropped_edges, self.picked_edges = self.adversarial_edges_drop(ori_trn_mat, self.adv_edges)
            
            trn_mat = ori_trn_mat               


        # trn_data = TrnData(ori_trn_mat)
        trn_data = TrnData(trn_mat)
        self.trn_loader = data.DataLoader(trn_data, batch_size=args.batch, shuffle=True, num_workers=0)

        tst_data = TstData(tst_mat, trn_mat)
        # tst_data = TstData(tst_mat, ori_trn_mat)
        self.tst_loader = data.DataLoader(tst_data, batch_size=args.tst_bat, shuffle=False, num_workers=0)

class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def neg_sampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                i_neg = np.random.randint(args.item)
                if (u, i_neg) not in self.dokmat:
                    break
            self.negs[i] = i_neg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
    def __init__(self, coomat, trn_mat):
        self.csrmat = (trn_mat.tocsr() != 0) * 1.0

        tst_locs = [None] * coomat.shape[0]
        tst_usrs = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            if tst_locs[row] is None:
                tst_locs[row] = list()
            tst_locs[row].append(col)
            tst_usrs.add(row)
        tst_usrs = np.array(list(tst_usrs))
        self.tst_usrs = tst_usrs
        self.tst_locs = tst_locs

    def __len__(self):
        return len(self.tst_usrs)

    def __getitem__(self, idx):
        return self.tst_usrs[idx], np.reshape(self.csrmat[self.tst_usrs[idx]].toarray(), [-1])

class temHandler:
    def __init__(self, adjs):
        self.torch_uni_adj, self.torch_adj = adjs

