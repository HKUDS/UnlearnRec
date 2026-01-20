import torch as t
import Utils.time_logger as logger
from Utils.time_logger import log
from params import args
from Model import *
from data_handler import DataHandler
import numpy as np
import pickle
import os
import setproctitle

class Coach:
    def __init__(self, handler):
        self.handler = handler

        print('NUM OF NODES', args.user + args.item)
        print('NUM OF EDGES', self.handler.trn_loader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Trn' + met] = list()
            self.metrics['Tst' + met] = list()
    
    def make_print(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepare_model()
        log('Model Prepared')        
        
        reses = self.tst_epoch(self.model)
        log(self.make_print('Topo', 0, reses, False))
        
        
    
    def prepare_model(self):
        # self.model = UnlearningMLP(self.handler).cuda()
        # self.model = LightGCN(self.handler).cuda()
        self.model = self.load_trained_model(args.trained_model)

    def tst_epoch(self, model):
        tst_loader = self.handler.tst_loader
        ep_recall, ep_ndcg = [0] * 2
        num = tst_loader.dataset.__len__()
        steps = num //args.tst_bat
        for i, tem in enumerate(tst_loader):
            usrs, trn_mask = tem
            usrs = usrs.long().cuda()
            trn_mask = trn_mask.cuda()

            all_preds = model.full_predict(usrs, trn_mask, self.handler.ts_ori_adj)
            _, top_locs = t.topk(all_preds, args.topk)
            recall, ndcg = self.cal_metrics(top_locs.cpu().numpy(), tst_loader.dataset.tst_locs, usrs)
            ep_recall += recall
            ep_ndcg += ndcg
            log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False, oneline=True)
        ret = dict()
        ret['Recall'] = ep_recall / num
        ret['NDCG'] = ep_ndcg / num
        return ret

    def cal_metrics(self, top_locs, tst_locs, bat_ids):
        assert top_locs.shape[0] == len(bat_ids)
        recall = ndcg = 0
        for i in range(len(bat_ids)):
            tem_top_locs = list(top_locs[i])
            tem_tst_locs = tst_locs[bat_ids[i]]
            tst_num = len(tem_tst_locs)
            max_dcg = np.sum([1 / (np.log2(loc + 2)) for loc in range(min(tst_num, args.topk))])
            tem_recall = dcg = 0
            for val in tem_tst_locs:
                if val in tem_top_locs:
                    tem_recall += 1
                    dcg += 1 / (np.log2(tem_top_locs.index(val) + 2))
            tem_recall /= tst_num
            tem_ndcg = dcg / max_dcg
            recall += tem_recall
            ndcg += tem_ndcg
        return recall, ndcg
    
    def save_history(self):
        if args.epoch == 0:
            return
        # with open('../../History/' + args.save_path + '.his', 'wb') as fs:
        #     pickle.dump(self.metrics, fs)

        content = {
            'model': self.model,
        }
        t.save(content,  args.save_path + '.mod')
        log('Model Saved: %s' % args.save_path)

    def load_model(self, load_model=args.load_model):
        ckp = t.load(load_model + '.mod')
        self.model = ckp['model']
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        

        # with open('../../History/' + load_model + '.his', 'rb') as fs:
        #     self.metrics = pickle.load(fs)
    def load_trained_model(self, trained_model = args.trained_model):
        ckp = t.load(trained_model + '.mod')
        print("####################tyep of trained model#####################")
        print(type(ckp))
        model = ckp['model']
        return model

    def test_unlearn(self, model, prefix=''):
        handler = self.handler
        unlearn_u, unlearn_i = handler.dropped_edges[0], handler.dropped_edges[1]

        print("####################unlearn_u, unlearn_i#####################")
        print(max(unlearn_u), min(unlearn_u), max(unlearn_i), min(unlearn_i))


        usr_embeds, itm_embeds =  model.forward( self.handler.ts_ori_adj)
        pretr_drp_res = innerProduct(usr_embeds[unlearn_u].detach(), itm_embeds[unlearn_i].detach())     

        pk_u, pk_i = handler.picked_edges[0], handler.picked_edges[1]     

        pretr_pos_res = innerProduct(usr_embeds[pk_u].detach(), itm_embeds[pk_i].detach())    


        drp_length = len(unlearn_u)
        pos_length = len(pk_u)
        rd_list = np.random.permutation(pos_length)
        pk_idx = rd_list[:drp_length]
        pk_u = t.tensor(pk_u).long()[pk_idx]
        pk_i = t.tensor(pk_i).long()[pk_idx]

        pretr_pos_pk_res = innerProduct(usr_embeds[pk_u].detach(), itm_embeds[pk_i].detach())    

        neg_rows, neg_cols = [], []
        rows, cols = self.handler.ori_trn_mat.row, self.handler.ori_trn_mat.col
        edge_set = set(list(map(lambda x: (rows[x], cols[x]), list(range(len(rows))))))
        for i in range(len(rows)):
            while True:
                rdm_row = np.random.randint(args.user)
                rdm_col = np.random.randint(args.item)
                if (rdm_row, rdm_col) not in edge_set:
                    edge_set.add((rdm_row, rdm_col))
                    break
            neg_rows.append(rdm_row)
            neg_cols.append(rdm_col)

        pretr_neg_res = innerProduct(usr_embeds[neg_rows].detach(), itm_embeds[neg_cols].detach())        
         


        print(f"#####################{prefix} picked & dropped difference###########################")
        print("pretr_drp_res:", pretr_drp_res)
        print("pretr_pos_res:", pretr_pos_res)
        print("pretr_pos_pk_res:", pretr_pos_pk_res)
        print("pretr_neg_res:", pretr_neg_res)
        
        print(f"Pretrain dropped edges scores : <{pretr_drp_res.mean().item()},{pretr_drp_res.var().item()},{pretr_drp_res.max().item()},{pretr_drp_res.min().item()}>")
        print(f"Pretrain positive edges scores: <{pretr_pos_res.mean().item()},{pretr_pos_res.var().item()},{pretr_pos_res.max().item()},{pretr_pos_res.min().item()}>")
        print(f"Pretrain picked positive edges scores: <{pretr_pos_pk_res.mean().item()},{pretr_pos_pk_res.var().item()},{pretr_pos_pk_res.max().item()},{pretr_pos_pk_res.min().item()}>")
        print(f"Pretrain negative edges scores: <{pretr_neg_res.mean().item()},{pretr_neg_res.var().item()},{pretr_neg_res.max().item()},{pretr_neg_res.min().item()}>")





if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.saveDefault = True

    print_args(args)
    log('Start')
    handler = DataHandler()
    handler.load_data(drop_rate=0.0, adv_attack=True)
    log('Load Data')

    coach = Coach(handler)
    coach.run()
