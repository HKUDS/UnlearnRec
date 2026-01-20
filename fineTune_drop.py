import torch as t
import Utils.time_logger as logger
from Utils.time_logger import log
from params import args
from Model import *
# from model import *

from data_handler import DataHandler
import numpy as np
import pickle
import os
import setproctitle
from Utils.utils import *
import random
import time
import itertools
from torch import nn

from copy import deepcopy

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
        # if args.load_model != None:
        #     self.load_model()
        #     self.model.is_training = False
        #     stloc = len(self.metrics['TrnLoss']) * args.tst_epoch - (args.tst_epoch - 1)
        #     log('Model Loaded')
        # else:
        #     stloc = 0
        #     log('Model Initialized')
        # reses = self.tst_epoch(self.model.topo_encoder)
        stloc = 0
        global_score_gap = -2e32
        self.handler.load_data(drop_rate=args.test_drop_rate, adv_attack=args.adversarial_attack)
        reses = self.tst_epoch(self.model)
        tot_time = 0
        log(self.make_print('Topo', 0, reses, False))
        for ep in range(stloc, args.epoch):
            # if ep % args.sim_epoch == 0:
            reses = self.tst_epoch(self.model)
            print(reses)    
            cur_gap = self.test_unlearn(self.model, prefix='Test without training:')
            # self.handler.load_data(drop_rate=args.drop_rate, adv_attack=False)
            if cur_gap > global_score_gap:
                self.save_history()
                global_score_gap = cur_gap

            

            # tst_flag = ep % args.tst_epoch == 0
            start = time.time()
            reses = self.trn_epoch()
            end = time.time()
            tot_time += (end - start)
            print("###############tot_time######################")
            print(tot_time)

            # log(self.make_print('Trn', ep, reses, tst_flag))
            self.learning_rate_decay()
            # if tst_flag:
            #     reses = self.tst_epoch(self.model)
            #     log(self.make_print('Tst', ep, reses, tst_flag))

                
            # print()
            print("##############parameters################")
            print(list(self.model.named_parameters()))
            print("##################ini_embeds in GraphUnlearning#######################")
            print(self.model.ini_embeds)
            print("##################fnl_embeds in GraphUnlearning#######################")
            print(self.model.fnl_embeds)
            



            # print("##############ini embeds################")
            # print(list(self.model.ini_embeds[:10]))

            # print("##############self.handler.ts_ori_adj################")
            # print(self.handler.ts_ori_adj)
            # print("##############self.handler.ts_drp_adj################")
            # print(self.handler.ts_drp_adj)

            print("##############self.handler.mask################")
            print(self.handler.mask)
            print(self.handler.mask.sum()/self.handler.mask.shape[0])

            # print("###################unlearn during training############################")
            # self.test_unlearn(self.model)

            # print("##############self.handler.ts_adj################")
            # print("##############parameters [1]################")
            # print(list(self.model.named_parameters())[1])

            # print("############## self.ini_embeds################")
            # print( self.model.ini_embeds)


        reses = self.tst_epoch(self.model)
        log(self.make_print('Tst', args.epoch, reses, True))

        # adjs = self.handler.random_drop_edges(rate=0.05)
        # self.model.unlearn(adjs)
        # reses = self.tst_epoch(self.model)
        # self.model.set_topo_encoder(handler)
        # log(self.make_print('Unlearn', args.epoch, reses, False))
        
        # self.save_history()
    
    def prepare_model(self):        
        self.model = self.load_model()

        trained_model = self.model.model
        trained_model.training = False
        ret = self.tst_epoch(trained_model, False)
        print("###################prepare_model test#########################")
        print(ret)

        # ini_embeds = trained_model.ini_embeds.detach()
        # ini_embeds = trained_model.ini_embeds.detach()
        # if hasattr(trained_model, "uEmbeds") and  hasattr(trained_model, "iEmbeds"):
        #     ini_embeds = t.concat([ trained_model.uEmbeds.detach() ,  trained_model.iEmbeds.detach()  ], axis=0).detach()
        #     ini_embeds.requires_grad=False
        # elif hasattr(trained_model, "ini_embeds"):
        #     ini_embeds = trained_model.ini_embeds.detach()
        #     ini_embeds.requires_grad=False
        
            
        # fnl_uEmbeds, fnl_iEmbeds = trained_model.forward(self.handler.ts_ori_adj) 
        # fnl_embeds = t.concat([fnl_uEmbeds.detach(), fnl_iEmbeds.detach()], axis=0).detach()

        fnl_uEmbeds, fnl_iEmbeds = trained_model.forward(self.handler.ts_pk_adj) 
        fnl_embeds = t.concat([fnl_uEmbeds.detach(), fnl_iEmbeds.detach()], axis=0).detach()
        self.model.model.fnl_embeds = fnl_embeds

        self.model.ini_embeds = nn.Parameter( deepcopy( self.model.ini_embeds.detach()))


        for name, params in self.model.named_parameters():
            if "mlp_layers" in name  or name == 'ini_embeds':
            # if  name == 'ini_embeds':
                print("############################para name #####################")
                print(name)
                params.requires_grad = True                
            else:
                params.requires_grad = False


        # self.model.model.is_training = False
        # ret = self.tst_epoch(self.model.model, False)
        # print("###################prepare_model test#########################")
        # print(ret)

        # fine_tune_params =  itertools.chain(self.model.mlp_layers.parameters(),  self.model.model.parameters())
        fine_tune_params =  self.model.parameters()

        self.opt = t.optim.Adam( fine_tune_params, lr=args.lr, weight_decay=0)
        print("##############parameters################")
        print(list(self.model.named_parameters()))
                
    
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
        ep_loss, ep_preloss, ep_unlearn_loss, ep_align_loss = [0] * 4
        steps = len(trn_loader)
        for i, tem in enumerate(trn_loader):
            # if i > 2500:
            #     steps = 2500
            #     break
            tem = list(map(lambda x: x.cuda(), tem))
            loss, loss_dict = self.model.cal_loss(tem, self.handler.ts_ori_adj, self.handler.ts_pk_adj, self.handler.mask,  self.handler.ts_drp_adj, self.handler.dropped_edges, self.handler.picked_edges )
            bpr_loss = loss_dict['bpr_loss']
            reg_loss = loss_dict['reg_loss']
            unlearn_loss = loss_dict['unlearn_loss']
            align_loss = loss_dict['align_loss']
            unlearn_ssl = loss_dict['unlearn_ssl']
            ep_loss += loss.item()
            ep_preloss += bpr_loss.item()
            ep_unlearn_loss += unlearn_loss.item()
            ep_align_loss += align_loss.item()



            self.opt.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm(self.model.parameters(), max_norm=1, norm_type=2)
            # print("#####################################self.model.edge_embeds1._grad.data########################################")
            # print(self.model.edge_embeds1._grad.data)
            # print("############################loss_dict after#################################")
            # print(loss_dict)

            # assert t.isnan(loss).sum() == 0, print(loss)
            # assert t.isnan(self.model.edge_embeds1._grad.data).sum() == 0, print("self.model.edge_embeds1._grad.data", self.model.edge_embeds1._grad.data)
            # assert t.isnan(self.model.uHyper._grad.data).sum() == 0, print("self.model.uHyper._grad.data", self.model.uHyper._grad.data)
            # assert t.isnan(self.model.iHyper._grad.data).sum() == 0, print("self.model.uHyper._grad.data", self.model.iHyper._grad.data)
            
            # self.model.param_train.grad[0].data.zero_()
            clear = self.handler.mask.reshape([-1]).tolist()
            if not args.allgrad:
                for i in range(len(clear)):
                    if clear[i] < 1-1e-6:
                        self.model.edge_embeds1._grad[i].data.zero_()
                        # self.model.edge_embeds2._grad[i].data.zero_()

                        # self.model.edge_embeds1.grad[i].data.zero_()
                        # self.model.edge_embeds2.grad[i].data.zero_()

                # self.model.edge_embeds1._grad[0].data.zero_()
                # self.model.edge_embeds2._grad[0].data.zero_()
                # print("#################self.model.param_train.grad#####################")
                # print(self.model.param_train.grad[li])

            # print("############################loss_dict before#################################")
            # print(loss_dict)
            self.opt.step()
            # print("###################self.model.param_train[0].grad####################")
            # print("############################loss_dict after#################################")
            # print(loss_dict)

            log('Step %d/%d: loss = %.6f, regLoss = %.6f, unlearn_loss = %.6f, align_loss = %.6f,unlearn_ssl = %.6f         ' % (i, steps, loss, reg_loss, unlearn_loss, align_loss, unlearn_ssl), save=False, oneline=True)
        ret = dict()
        ret['Loss'] = ep_loss / steps
        ret['preLoss'] = ep_preloss / steps
        ret['unlearn_loss'] = ep_unlearn_loss/ steps
        ret['align_loss'] = ep_align_loss/ steps
        return ret

    def tst_epoch(self, model, unlearn_flag=True):
        tst_loader = self.handler.tst_loader
        ep_recall, ep_ndcg = [0] * 2
        num = tst_loader.dataset.__len__()
        steps = num //args.tst_bat
        for i, tem in enumerate(tst_loader):
            usrs, trn_mask = tem
            usrs = usrs.long().cuda()
            trn_mask = trn_mask.cuda()
            if unlearn_flag:
                all_preds = model.full_predict(self.handler.ts_ori_adj, self.handler.ts_pk_adj, self.handler.mask,  self.handler.ts_drp_adj, usrs, trn_mask)
            else:                
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
    

    def test_unlearn(self, model, prefix=''):
        handler = self.handler
        unlearn_u, unlearn_i = handler.dropped_edges[0], handler.dropped_edges[1]
        usr_embeds, itm_embeds =  model.outforward( handler.ts_ori_adj, handler.ts_pk_adj ,handler.mask, handler.ts_drp_adj)
        our_drp_res = innerProduct(usr_embeds[unlearn_u].detach(), itm_embeds[unlearn_i].detach())     

        # pretr_u_emb, pretr_i_emb = model.fnl_embeds[:args.user].detach(), model.fnl_embeds[args.user:].detach()
        # pretr_drp_res = innerProduct(pretr_u_emb[unlearn_u].detach(), pretr_i_emb[unlearn_i].detach())

        pretr_u_emb, pretr_i_emb = model.model.forward(self.handler.ts_ori_adj)
        pretr_u_emb, pretr_i_emb = pretr_u_emb.detach(), pretr_i_emb.detach()
        pretr_drp_res = innerProduct(pretr_u_emb[unlearn_u].detach(), pretr_i_emb[unlearn_i].detach())

        pk_u, pk_i = handler.picked_edges[0], handler.picked_edges[1]     

        drp_length = len(unlearn_u)
        pos_length = len(pk_u)
        rd_list = np.random.permutation(pos_length)
        pk_idx = rd_list[:drp_length]
        pk_u = t.tensor(pk_u).long()[pk_idx]
        pk_i = t.tensor(pk_i).long()[pk_idx]

        our_pos_res = innerProduct(usr_embeds[pk_u].detach(), itm_embeds[pk_i].detach())
        pretr_pos_res = innerProduct(pretr_u_emb[pk_u].detach(), pretr_i_emb[pk_i].detach())


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

        our_neg_res = innerProduct(usr_embeds[neg_rows].detach(), itm_embeds[neg_cols].detach())        
        pretr_neg_res = innerProduct(pretr_u_emb[neg_rows].detach(), pretr_i_emb[neg_cols].detach())



        print(f"#####################{prefix} unlearning effacy###########################")
        # print(our_drp_res, pretr_drp_res, our_pos_res)
        # print(our_drp_res, pretr_drp_res, our_pos_res)
        print("our_drp_res:", our_drp_res)
        print("our_pos_res:", our_pos_res)
        print("pretr_drp_res:", pretr_drp_res)
        print("pretr_pos_res:", pretr_pos_res)
        print("our_neg_res:", our_neg_res)
        print("pretr_neg_res:", pretr_neg_res)


        # print(our_drp_res.mean().item(), pretr_drp_res.mean().item(), our_pos_res.mean().item())
        print(f"Our dropped edges scores (mean, var, max, min): <{our_drp_res.mean().item()}, {our_drp_res.var().item()}, {our_drp_res.max().item()}, {our_drp_res.min().item()}>,  Pretrain dropped edges scores:<{pretr_drp_res.mean().item()},{pretr_drp_res.var().item()},{pretr_drp_res.max().item()},{pretr_drp_res.min().item()}>")
        print(f"Our positive edges scores(mean, var, max, min): <{our_pos_res.mean().item()}, {our_pos_res.var().item()}, {our_pos_res.max().item()}, {our_pos_res.min().item()}>,  Pretrain positive edges scores:<{pretr_pos_res.mean().item()},{pretr_pos_res.var().item()},{pretr_pos_res.max().item()},{pretr_pos_res.min().item()}>")
        print(f"Our Negative edges scores(mean, var, max, min): <{our_neg_res.mean().item()}, {our_neg_res.var().item()}, {our_neg_res.max().item()}, {our_neg_res.min().item()}>,  Pretrain negative edges scores:<{pretr_neg_res.mean().item()},{pretr_neg_res.var().item()},{pretr_neg_res.max().item()},{pretr_neg_res.min().item()}>")

        return  our_neg_res.mean().item() - our_drp_res.mean().item()


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

    def load_trained_model(self, trained_model = args.trained_model):
        # ckp = t.load(trained_model + '.mod')
        ckp = t.load(trained_model)
        print("####################tyep of trained model#####################")
        print(type(ckp))
        model = ckp['model']
        return model
        

    def load_model(self, load_model=args.load_model):
        ckp = t.load(load_model + '.mod')
        self.model = ckp['model']
        # self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        return self.model

        # with open('../../History/' + load_model + '.his', 'rb') as fs:
        #     self.metrics = pickle.load(fs)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.saveDefault = True

    print_args(args)
    t.manual_seed(args.seed)
    t.cuda.manual_seed_all(args.seed)
    t.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)

    log('Start')
    handler = DataHandler()
    handler.load_data(drop_rate=args.test_drop_rate, adv_attack=args.adversarial_attack)
    # handler.load_data(drop_rate=args.drop_rate)
    log('Load Data')

    coach = Coach(handler)
    coach.run()
