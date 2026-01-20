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
        if args.load_model != None:
            self.load_model()
            self.model.is_training = False
            stloc = len(self.metrics['TrnLoss']) * args.tst_epoch - (args.tst_epoch - 1)
            log('Model Loaded')
        else:
            stloc = 0
            log('Model Initialized')
        # reses = self.tst_epoch(self.model.topo_encoder)
        reses = self.tst_epoch(self.model)
        log(self.make_print('Topo', 0, reses, False))
        for ep in range(stloc, args.epoch):
            tst_flag = ep % args.tst_epoch == 0
            reses = self.trn_epoch()
            log(self.make_print('Trn', ep, reses, tst_flag))
            self.learning_rate_decay()
            if tst_flag:
                reses = self.tst_epoch(self.model)
                log(self.make_print('Tst', ep, reses, tst_flag))

                # adjs = self.handler.random_drop_edges(rate=args.unlearn_rate)
                # self.model.unlearn(adjs)
                # reses = self.tst_epoch(self.model)
                # self.model.set_topo_encoder(handler)
                # log(self.make_print('Unlearn', ep, reses, False))

                self.save_history()

        reses = self.tst_epoch(self.model)
        log(self.make_print('Tst', args.epoch, reses, True))

        # adjs = self.handler.random_drop_edges(rate=0.05)
        # self.model.unlearn(adjs)
        # reses = self.tst_epoch(self.model)
        # self.model.set_topo_encoder(handler)
        # log(self.make_print('Unlearn', args.epoch, reses, False))
        
        self.save_history()
    
    def prepare_model(self):
        # self.model = UnlearningMLP(self.handler).cuda()
        self.model = LightGCN(self.handler).cuda()

        # for name, params in self.model.named_parameters():
        #     if name=="param_fixed":
        #         params.requires_grad = False

        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        # print("##############parameters################")
        # print(list(self.model.named_parameters()))
        
        
    
    def learning_rate_decay(self):
        if args.decay == 1.0:
            return
        for param_group in self.opt.param_groups:
            lr = param_group['lr'] * args.decay
            if lr > 1e-4:
                param_group['lr'] = lr
        return
    
    def trn_epoch(self):
        trn_loader = self.handler.trn_loader
        trn_loader.dataset.neg_sampling()
        ep_loss, ep_preloss = [0] * 2
        steps = len(trn_loader)
        for i, tem in enumerate(trn_loader):
            if i > 2500:
                steps = 2500
                break
            tem = list(map(lambda x: x.cuda(), tem))
            loss, loss_dict = self.model.cal_loss(tem)
            bpr_loss = loss_dict['bpr_loss']
            reg_loss = loss_dict['reg_loss']
            ep_loss += loss.item()
            ep_preloss += bpr_loss.item()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            log('Step %d/%d: loss = %.6f, regLoss = %.6f         ' % (i, steps, loss, reg_loss), save=False, oneline=True)
        ret = dict()
        ret['Loss'] = ep_loss / steps
        ret['preLoss'] = ep_preloss / steps
        return ret

    def tst_epoch(self, model):
        tst_loader = self.handler.tst_loader
        ep_recall, ep_ndcg = [0] * 2
        num = tst_loader.dataset.__len__()
        steps = num //args.tst_bat
        for i, tem in enumerate(tst_loader):
            usrs, trn_mask = tem
            usrs = usrs.long().cuda()
            trn_mask = trn_mask.cuda()

            all_preds = model.full_predict(usrs, trn_mask)
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

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.saveDefault = True

    log('Start')
    handler = DataHandler()
    handler.load_data()
    log('Load Data')

    coach = Coach(handler)
    coach.run()