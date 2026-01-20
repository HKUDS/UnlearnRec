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
        if args.fineTune :
            self.load_model_2_finetune()
            self.model.is_training = False
            stloc = len(self.metrics['TrnLoss']) * args.tst_epoch - (args.tst_epoch - 1)
            log('Model Loaded')
        else:
            stloc = 0
            log('Model Initialized')
        # reses = self.tst_epoch(self.model.topo_encoder)
        global_score_gap = -2e32

        topo_reses = self.tst_epoch(self.model)

        log(self.make_print('Topo', 0, topo_reses, False))
        for ep in range(stloc, args.epoch):
            if ep % args.sim_epoch == 0:
                self.handler.load_data(drop_rate=args.test_drop_rate, adv_attack=args.adversarial_attack)
                # reses = self.trn_epoch()
                reses = self.tst_epoch(self.model)                
                print(f">>>>Recall: {reses['Recall']:.6f},  NDCG: {reses['NDCG']:.6f} @ Epoch: {ep}")
    

                assert reses['Recall'] >= topo_reses['Recall'] * args.perf_degrade, "The prediction performance has been lower than requirements, you can increase align_wei, decrease perf_degrade and tune other parameters"

                cur_gap = self.test_unlearn(self.model, prefix='Test without training:')
                
                self.handler.load_data(drop_rate=args.pretrain_drop_rate, adv_attack=False)

                if cur_gap > global_score_gap:
                    self.save_history()
                    global_score_gap = cur_gap



            tst_flag = ep % args.tst_epoch == 0
            reses = self.trn_epoch()
            log(self.make_print('Trn', ep, reses, tst_flag))
            self.learning_rate_decay()
            if tst_flag:
                reses = self.tst_epoch(self.model)
                log(self.make_print('Tst', ep, reses, tst_flag))
                assert reses['Recall'] >= topo_reses['Recall'] * args.perf_degrade, "The prediction performance has been lower than requirements, you can increase align_wei, decrease perf_degrade and tune other parameters"

                
            # ######################################Debug############################################
            # print("##############parameters################")
            # print(list(self.model.named_parameters()))
            # print("##################ini_embeds in GraphUnlearning#######################")
            # print(self.model.ini_embeds)
            # print("##################fnl_embeds in GraphUnlearning#######################")
            # print(self.model.fnl_embeds)
            # print("##############self.handler.mask################")
            # print(self.handler.mask)
            # print(self.handler.mask.sum()/self.handler.mask.shape[0])
            # #######################################################################################
            
        reses = self.tst_epoch(self.model)
        log(self.make_print('Tst', args.epoch, reses, True))

    
    def prepare_model(self):
        trained_model = self.load_trained_model()
        trained_model.is_training = False
        ret = self.tst_epoch(trained_model, False)
        # ######################################Debug############################################
        # print("###################prepare_model test#########################")
        # print(ret)
        # #######################################################################################

        if hasattr(trained_model, "uEmbeds") and  hasattr(trained_model, "iEmbeds"):
            ini_embeds = t.concat([ trained_model.uEmbeds.detach() ,  trained_model.iEmbeds.detach()  ], axis=0).detach()
            ini_embeds.requires_grad=False
        elif hasattr(trained_model, "ini_embeds"):
            ini_embeds = trained_model.ini_embeds.detach()
            ini_embeds.requires_grad=False
        
            
        fnl_uEmbeds, fnl_iEmbeds = trained_model.forward(self.handler.ts_ori_adj, keepRate=1.0) 
        fnl_embeds = t.concat([fnl_uEmbeds.detach(), fnl_iEmbeds.detach()], axis=0).detach()

        self.model = GraphUnlearning(self.handler, trained_model,  ini_embeds.detach(), fnl_embeds.detach() ).cuda()

        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)


    
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
            if i > 2500:
                steps = 2500
                break
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
            ep_align_loss += align_loss



            self.opt.zero_grad()
            loss.backward()

            assert t.isnan(loss).sum() == 0, print(loss)
            assert t.isnan(self.model.edge_embeds1._grad.data).sum() == 0, print("self.model.edge_embeds1._grad.data", self.model.edge_embeds1._grad.data)
            # assert t.isnan(self.model.uHyper._grad.data).sum() == 0, print("self.model.uHyper._grad.data", self.model.uHyper._grad.data)
            # assert t.isnan(self.model.iHyper._grad.data).sum() == 0, print("self.model.uHyper._grad.data", self.model.iHyper._grad.data)
            
            # self.model.param_train.grad[0].data.zero_()
            clear = self.handler.mask.reshape([-1]).tolist()
            if not args.allgrad:
                for i in range(len(clear)):
                    if clear[i] < 1-1e-6:
                        self.model.edge_embeds1._grad[i].data.zero_()
     
                # print("#################self.model.param_train.grad#####################")
                # print(self.model.param_train.grad[li])

            # print("############################loss_dict before#################################")
            # print(loss_dict)
            self.opt.step()


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

        pretr_u_emb, pretr_i_emb = model.model.forward(self.handler.ts_ori_adj,  keepRate=1.0)
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



        # #############################################debug#############################################
        # print(f"#####################{prefix} unlearning effacy###########################")
        # # print(our_drp_res, pretr_drp_res, our_pos_res)
        # # print(our_drp_res, pretr_drp_res, our_pos_res)
        # print("our_drp_res:", our_drp_res)
        # print("our_pos_res:", our_pos_res)
        # print("pretr_drp_res:", pretr_drp_res)
        # print("pretr_pos_res:", pretr_pos_res)
        # print("our_neg_res:", our_neg_res)
        # print("pretr_neg_res:", pretr_neg_res)
        # print(our_drp_res.mean().item(), pretr_drp_res.mean().item(), our_pos_res.mean().item())
        # ##############################################################################################
        
        print("##############################################################Pretraining For Unlearning Monitor##########################################################################")
        print(f"Pretraining for Unlearning: Simulated Dropped edges scores  (mean, var, max, min): <{our_drp_res.mean().item():0.4f}, {our_drp_res.var().item():0.4f}, {our_drp_res.max().item():0.4f}, {our_drp_res.min().item():0.4f}>   |    Before unlearning: Current Dropped edges scores:   <{pretr_drp_res.mean().item():0.4f},{pretr_drp_res.var().item():0.4f},{pretr_drp_res.max().item():0.4f},{pretr_drp_res.min().item():0.4f}>")
        print(f"Pretraining for Unlearning: Positive edges scores (mean, var, max, min):           <{our_pos_res.mean().item():0.4f}, {our_pos_res.var().item():0.4f}, {our_pos_res.max().item():0.4f}, {our_pos_res.min().item():0.4f}>   |    Before unlearning: Positive edges scores:          <{pretr_pos_res.mean().item():0.4f},{pretr_pos_res.var().item():0.4f},{pretr_pos_res.max().item():0.4f},{pretr_pos_res.min().item():0.4f}>")
        print(f"Pretraining for Unlearning: Negative edges scores (mean, var, max, min):           <{our_neg_res.mean().item():0.4f}, {our_neg_res.var().item():0.4f}, {our_neg_res.max().item():0.4f}, {our_neg_res.min().item():0.4f}>   |    Before unlearning: Negative edges scores:          <{pretr_neg_res.mean().item():0.4f},{pretr_neg_res.var().item():0.4f},{pretr_neg_res.max().item():0.4f},{pretr_neg_res.min().item():0.4f}>")
        print("##########################################################################################################################################################################")
        
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

        content = {
            'model': self.model,
        }
        save_path = args.save_path
        if not save_path.endswith('.mod'):
            save_path = save_path + '.mod'

        t.save(content,  save_path)
        log('Model Saved: %s' % args.save_path)

    def load_trained_model(self, trained_model = args.trained_model):
        if not trained_model.endswith('.mod'):
            trained_model = trained_model + '.mod'

        ckp = t.load(trained_model)

        model = ckp['model']
        return model
        

    def load_model_2_finetune(self, model_2_finetune=args.model_2_finetune):
        if not model_2_finetune.endswith('.mod'):
            model_2_finetune = model_2_finetune + '.mod'

        ckp = t.load(model_2_finetune)
        self.model = ckp['model']
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.saveDefault = True

    print_args(args)
    t.manual_seed(args.seed)
    t.cuda.manual_seed_all(args.seed)
    t.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.fineTune = False

    log('Start')
    handler = DataHandler()
    handler.load_data(drop_rate=args.pretrain_drop_rate, adv_attack=False)
    # handler.load_data(drop_rate=args.drop_rate)
    log('Load Data')

    coach = Coach(handler)
    coach.run()
