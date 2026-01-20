from statistics import mean
import torch as t
from torch import nn
import torch.nn.functional as F
from params import args
from Utils.utils import *
import numpy as np
import scipy
import torch_sparse as ts
from data_handler import temHandler

init = nn.init.xavier_uniform_
uniform_init = nn.init.uniform_


import networkx as nx



class SpanningTree(nn.Module):
    def __init__(self, adj):
        super(SpanningTree, self).__init__()
        self.old_adj = adj

    def to_graph_list(self, adj):
        self.graph_list = []
        rows, cols, vals = adj.coo()    
        for i in range(rows.shape[0]):
            r = rows[i].item()
            c = cols[i].item()
            v = vals[i].item()
            if r <= c:
                self.graph_list.append((r,c,v))
        return self.graph_list

    def to_sparse_adj(self, shape, T):
        rows = []
        cols = []
        vals = []

        for tup in T:
            rows.append(tup[0])
            cols.append(tup[1])
            vals.append(tup[2]['weight'])

            rows.append(tup[1])
            cols.append(tup[0])
            vals.append(tup[2]['weight'])
        rows = t.tensor(rows)
        cols = t.tensor(cols)
        vals = t.tensor(vals)

        return ts.SparseTensor(row=rows, col=cols, value = vals, sparse_sizes= shape).cuda()

    def forward(self, adj):
        if adj == self.old_adj:
            return self.new_adj

        self.old_adj = adj
        self.to_graph_list(adj)
        G = nx.Graph()
        G.add_weighted_edges_from(self.graph_list)  
        T = nx.minimum_spanning_tree(G)    
        self.new_adj =  self.to_sparse_adj(adj.sizes(), T.edges(data=True))

        return self.new_adj



class SpAdjDropEdge(nn.Module):
    def __init__(self):
        super(SpAdjDropEdge, self).__init__()

    def forward(self, adj, keepRate):
        if keepRate == 1.0:
            return adj

        row, col, val = adj.coo()
        edgeNum = val.size()

        mask = ((t.rand(edgeNum) + keepRate).floor()).type(t.bool)
        newVals = val[mask] / keepRate  #  v1
        # newVals = val[mask]   #  v2

        newRow = row[mask]
        newCol = col[mask]

        return ts.SparseTensor(row=newRow, col=newCol, value = newVals, sparse_sizes= adj.sizes())


class HGNNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=False, act=None):
        super(HGNNLayer, self).__init__()
        # self.act = nn.LeakyReLU(negative_slope=args.leaky)
        self.W1 = nn.Parameter(t.eye(in_feat, out_feat).cuda() )
        self.bias1 = nn.Parameter(t.zeros( 1, out_feat).cuda() )

        self.W2 = nn.Parameter(t.eye(out_feat, in_feat).cuda())
        self.bias2 = nn.Parameter(t.zeros( 1, in_feat).cuda())


        if act == 'identity' or act is None:
            self.act = None
        elif act == 'leaky':
            self.act = nn.LeakyReLU(negative_slope=args.leaky)
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'relu6':
            self.act = nn.ReLU6()
        else:
            raise Exception('Error')

    def forward(self, embeds):
        # if self.act is None:
        #     # return self.linear(embeds)
        #     return  embeds @ self.W 
        out1 = self.act(  embeds @ self.W1 + self.bias1 )
        out2 = self.act(  out1 @ self.W2 + self.bias2  )
        return out2


    
    
class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()
		# self.act = nn.LeakyReLU(negative_slope=args.leaky)

	def forward(self, adj, embeds):
		# return (t.spmm(adj, embeds))
		return adj.matmul(embeds)


class GraphUnlearning(nn.Module):
    def __init__(self, handler, model, ini_embeds, fnl_embeds):
        super(GraphUnlearning, self).__init__()        
        edges_num = handler.ts_ori_adj.nnz()
        self.edge_embeds1 = nn.Parameter(t.zeros(args.user+ args.item, args.latdim).cuda())
        # self.edge_embeds2 = nn.Parameter(t.zeros(args.user+ args.item, args.latdim).cuda())

        self.mlp_layers = nn.Sequential(*[FeedForwardLayer(args.latdim, args.latdim, act=args.act) for i in range(args.layer_mlp)])
        # self.layer_norm = nn.LayerNorm(args.latdim)
        self.act = nn.LeakyReLU(negative_slope=args.leaky)

        if args.withdraw_rate_init == 1:
            self.withdraw_rate = nn.Parameter(t.ones(args.user+args.item, 1) * args.lr * 2)            
        else:
            self.withdraw_rate = nn.Parameter(t.zeros(args.user+args.item, 1) * args.lr * 10)

        self.edgeDropper = SpAdjDropEdge()
        self.gcnLayer = GCNLayer()

        self.ini_embeds = ini_embeds.detach()
        self.fnl_embeds = fnl_embeds.detach()

        self.ini_embeds.requires_grad = False
        self.fnl_embeds.requires_grad = False
        self.model  = model


        if hasattr(self.model, "uEmbeds") and  hasattr(self.model, "iEmbeds"):  
            self.model.uEmbeds.detach()
            self.model.uEmbeds.requires_grad = False    
            self.model.iEmbeds.detach()
            self.model.iEmbeds.requires_grad = False                
        else:
            self.model.ini_embeds.detach()
            self.model.ini_embeds.requires_grad = False            


    def forward(self, ori_adj, ts_pk_adj ,mask, ts_drp_adj):        
        lats = [self.edge_embeds1 ]
        gnnLats = []        
        hyperLats = []           

        for _ in range(args.gnn_layer):            
            temEmbeds = self.gcnLayer(self.edgeDropper(ts_drp_adj, 1.0), lats[-1])
            hyperemb = self.gcnLayer(self.edgeDropper(ts_drp_adj, 0.95), lats[-1])

            gnnLats.append(temEmbeds)
            hyperLats.append(hyperemb)
            lats.append(  temEmbeds )
        edge_embed = sum(lats)
                
        edges_embeddings = [edge_embed]
        for _ in range(args.unlearn_layer):
            edges_embeddings.append(ts_pk_adj.matmul(edges_embeddings[-1]))
        
        withdraw =  [ self.fnl_embeds * self.withdraw_rate ]        
        for _ in range(args.gnn_layer):
            withdraw.append(ts_drp_adj.matmul(withdraw[-1]))

        delta_emb = - args.overall_withdraw_rate* withdraw[-1] +  edges_embeddings[-1]
        
        for i, layer in enumerate(self.mlp_layers):
            delta_emb = layer(delta_emb)

        tuned_emb = self.ini_embeds + delta_emb

        return tuned_emb, gnnLats, hyperLats  

    def outforward(self, ori_adj, ts_pk_adj ,mask, ts_drp_adj):
        self.model.training = False
        tuned_emb , _ , _ = self.forward(ori_adj, ts_pk_adj ,mask, ts_drp_adj)
        out_emb = self.model.forward(ts_pk_adj, tuned_emb, keepRate=1.0)        
        usr_embeds, itm_embeds = out_emb[:2]
        
        return usr_embeds, itm_embeds
    

    def out_all_layer(self, ori_adj, ts_pk_adj ,mask, ts_drp_adj, layer=0):
        self.model.training = False
        tuned_emb, _, _ = self.forward(ori_adj, ts_pk_adj ,mask, ts_drp_adj)
        all_embs, out_emb = self.model.forward(ts_pk_adj, tuned_emb, all_layer=True)  

        if layer == -2:
            tuned_emb[:args.user], tuned_emb[args.user:]
        elif layer == -1:
            return out_emb[:args.user], out_emb[args.user:]
        else:
            return out_emb[layer][:args.user], out_emb[layer][args.user:]
            

    def cal_loss(self, batch_data, ori_adj, ts_pk_adj , mask, ts_drp_adj, drp_edges,  pk_edges=None):
        ancs, poss, negs = list(map(lambda x: x.long(), batch_data))

        self.model.training = True
        tuned_emb,  gcnEmbedsLst, hyperEmbedsLst  = self.forward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        out_emb = self.model.forward(ts_pk_adj, tuned_emb)
        
        usr_embeds, itm_embeds = out_emb[:2]


        base_loss, loss_dict =  self.model.cal_loss(batch_data,  tuned_emb=tuned_emb , ori_adj=ori_adj, ts_pk_adj=ts_pk_adj , mask=mask, ts_drp_adj=ts_drp_adj, drp_edges=drp_edges,  pk_edges=None)

        if args.unlearn_type =='v1':
            unlearn_loss = cal_neg_aug_v1(usr_embeds[drp_edges[0]],  itm_embeds[drp_edges[1]])
        elif args.unlearn_type =='v2':
            unlearn_loss = cal_neg_aug_v2(usr_embeds[drp_edges[0]],  itm_embeds[drp_edges[1]])

        tar_fnl_uEmbeds, tar_fnl_iEmbeds  = self.fnl_embeds[ :args.user].detach(), self.fnl_embeds[args.user: ].detach()
            
        if args.fineTune:
            loss_dict['unlearn_loss'] = unlearn_loss
            # loss_dict['align_loss'] = align_loss
            align_loss= cal_positive_pred_align_v2(usr_embeds[ancs], tar_fnl_uEmbeds[ancs], itm_embeds[poss], tar_fnl_iEmbeds[poss],   cal_l2_distance, temp=args.align_temp)
            loss_dict['align_loss'] =  align_loss
            loss_dict['unlearn_ssl'] = t.tensor(0.)
            return base_loss + args.unlearn_wei * unlearn_loss + args.align_wei*align_loss ,    loss_dict
            # return base_loss + args.unlearn_wei * unlearn_loss  ,    loss_dict


        

        if not args.fineTune:
            if args.align_type == 'v2':
                align_loss = cal_positive_pred_align_v2(usr_embeds[ancs], tar_fnl_uEmbeds[ancs], itm_embeds[poss], tar_fnl_iEmbeds[poss],   cal_l2_distance, temp=args.align_temp)
            elif args.align_type == 'v3':
                align_loss = cal_positive_pred_align_v3(usr_embeds[ancs], tar_fnl_uEmbeds[ancs], itm_embeds[poss], tar_fnl_iEmbeds[poss],   cal_l2_distance, temp=args.align_temp)        

        sslLoss = 0
        for i in range(args.gnn_layer):
            embeds1 = gcnEmbedsLst[i].detach()
            embeds2 = hyperEmbedsLst[i]
            sslLoss += contrastLoss(embeds1[:args.user], embeds2[:args.user], t.unique(ancs), args.hyper_temp) + contrastLoss(embeds1[args.user:], embeds2[args.user:], t.unique(poss), args.hyper_temp)
  
        loss = args.unlearn_wei * unlearn_loss  + args.align_wei*align_loss + base_loss + args.unlearn_ssl*sslLoss
        loss_dict['unlearn_loss'] = unlearn_loss
        loss_dict['align_loss'] = align_loss
        loss_dict['unlearn_ssl'] = sslLoss
                

        return loss, loss_dict

    # (self.handler.ts_ori_adj, self.handler.ts_pk_adj, self.handler.mask,  self.handler.ts_drp_adj, usrs, trn_mask)
    def full_predict(self, ori_adj, ts_pk_adj, mask, ts_drp_adj , usrs, trn_mask):

        self.model.training = False
        usr_embeds, itm_embeds = self.outforward(ori_adj, ts_pk_adj, mask, ts_drp_adj)

        pck_usr_embeds = usr_embeds[usrs]
        full_preds = pck_usr_embeds @ itm_embeds.T
        full_preds = full_preds * (1 - trn_mask) - trn_mask * 1e8
        return full_preds

class LightGCN(nn.Module):
    def __init__(self, handler):
        super(LightGCN, self).__init__()

        # self.adj = handler.torch_adj
        self.handler = handler
        self.adj = handler.ts_ori_adj
        self.ini_embeds = nn.Parameter(init(t.empty(args.user + args.item, args.latdim)))

    def forward(self, adj, ini_embeds=None, all_layer=False, keepRate=None):
        if ini_embeds is None:
            ini_embeds = self.ini_embeds

        embedsList = [ini_embeds]
        for _ in range(args.gnn_layer):
            embedsList.append(adj.matmul(embedsList[-1]))
        embeds = sum(embedsList)

        if all_layer:
            return (embedsList, embeds)

        return embeds[:args.user], embeds[args.user:]
    
    def cal_loss(self, batch_data,  tuned_emb=None , ori_adj=None, ts_pk_adj=None , mask=None, ts_drp_adj=None, drp_edges=None,  pk_edges=None):
        ancs, poss, negs = list(map(lambda x: x.long(), batch_data))
        usr_embeds, itm_embeds = self.forward( ts_pk_adj , tuned_emb)
        bpr_loss = cal_bpr(usr_embeds[ancs], itm_embeds[poss], itm_embeds[negs]) * args.bpr_wei

        reg_loss = cal_reg(self) * args.reg        
        loss = bpr_loss + reg_loss 
    
        loss_dict = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss}
        return loss, loss_dict

    def full_predict(self, usrs, trn_mask, adj):
        usr_embeds, itm_embeds = self.forward(adj)
        pck_usr_embeds = usr_embeds[usrs]
        full_preds = pck_usr_embeds @ itm_embeds.T
        full_preds = full_preds * (1 - trn_mask) - trn_mask * 1e8
        return full_preds





init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class SimGCL(nn.Module):
    def __init__(self, handler):
        super(SimGCL, self).__init__()
        
        self.adj = handler.ts_ori_adj
        self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
        self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
        self.gcnLayers = nn.Sequential(*[SimGclGCNLayer() for i in range(args.gnn_layer)])
        self.perturbGcnLayers1 = nn.Sequential(*[SimGclGCNLayer(perturb=True) for i in range(args.gnn_layer)])
        self.perturbGcnLayers2 = nn.Sequential(*[SimGclGCNLayer(perturb=True) for i in range(args.gnn_layer)])


    def getEgoEmbeds(self, adj):
        uEmbeds, iEmbeds = self.forward(adj)
        return t.concat([uEmbeds, iEmbeds], axis=0)

    def forward(self, adj, iniEmbeds=None, all_layer=False, keepRate=None):
        if iniEmbeds is None:          
            iniEmbeds = t.concat([self.uEmbeds, self.iEmbeds], axis=0)        
                                            
        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        mainEmbeds = sum(embedsLst[1:]) / len(embedsLst[1:])
        if all_layer:
            return (embedsLst,mainEmbeds)

        if self.training:
            perturbEmbedsLst1 = [iniEmbeds]
            for gcn in self.perturbGcnLayers1:
                embeds = gcn(adj, perturbEmbedsLst1[-1])
                perturbEmbedsLst1.append(embeds)
            perturbEmbeds1 = sum(perturbEmbedsLst1[1:]) / len(embedsLst[1:])

            perturbEmbedsLst2 = [iniEmbeds]
            for gcn in self.perturbGcnLayers2:
                embeds = gcn(adj, perturbEmbedsLst2[-1])
                perturbEmbedsLst2.append(embeds)
            perturbEmbeds2 = sum(perturbEmbedsLst2[1:]) / len(embedsLst[1:])

            return mainEmbeds[:args.user], mainEmbeds[args.user:], perturbEmbeds1[:args.user], perturbEmbeds1[args.user:], perturbEmbeds2[:args.user], perturbEmbeds2[args.user:]
        return mainEmbeds[:args.user], mainEmbeds[args.user:]

    def cal_loss(self, batch_data,  tuned_emb=None , ori_adj=None, ts_pk_adj=None , mask=None, ts_drp_adj=None, drp_edges=None,  pk_edges=None):
        ancs, poss, negs = list(map(lambda x: x.long(), batch_data))
        # ancs, poss, negs = tem
        ancs = ancs.long().cuda()
        poss = poss.long().cuda()
        negs = negs.long().cuda()
        self.train()
        # print("###################cal_loss self.train########################")
        # print(self.training)
        usrEmbeds, itmEmbeds, pUsrEmbeds1, pItmEmbeds1, pUsrEmbeds2, pItmEmbeds2 = self.forward(ts_pk_adj, tuned_emb )

        ancEmbeds = usrEmbeds[ancs]
        posEmbeds = itmEmbeds[poss]
        negEmbeds = itmEmbeds[negs]

        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        bprLoss = - (scoreDiff).sigmoid().log().mean()
        if args.reg_version == 'v1':
            regLoss = SimGCL_calcRegLoss(ancEmbeds, posEmbeds) 
        elif args.reg_version == 'v2':
            regLoss = SimGCL_calcRegLoss_v2(ancEmbeds, posEmbeds) 
        else:
            regLoss = SimGCL_calcRegLoss_v3(ancEmbeds, posEmbeds)                 

        contrastLoss = (contrast(pUsrEmbeds1, pUsrEmbeds2, ancs, args.temp) + contrast(pItmEmbeds1, pItmEmbeds2, poss, args.temp)) 
        # contrastLoss = 0


        loss = args.bpr_wei * bprLoss +  args.reg * regLoss + args.ssl_reg * contrastLoss   

        loss_dict = {'bpr_loss': bprLoss, 'reg_loss': regLoss, "contrast_loss": contrastLoss}
        return loss, loss_dict                

    def full_predict(self, usrs, trn_mask, adj):
        self.training = False
        usr_embeds, itm_embeds = self.forward(adj)
        pck_usr_embeds = usr_embeds[usrs]
        full_preds = pck_usr_embeds @ itm_embeds.T
        full_preds = full_preds * (1 - trn_mask) - trn_mask * 1e8
        return full_preds


class SimGclGCNLayer(nn.Module):
    def __init__(self, perturb=False):
        super(SimGclGCNLayer, self).__init__()
        self.perturb = perturb

    def forward(self, adj, embeds):
        # ret = t.spmm(adj, embeds)
        ret = adj.matmul(embeds)
        if not self.perturb:
            return ret
        # noise = (F.normalize(t.rand(ret.shape).cuda(), p=2) * t.sign(ret)) * args.eps
        random_noise = t.rand_like(ret).cuda()
        noise = t.sign(ret) * F.normalize(random_noise, dim=-1) * args.eps
        return ret + noise


def get_shape(adj):
    if isinstance(adj, ts.SparseTensor):
        return adj.sizes()
    else:
        return adj.shape


class FeedForwardLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=False, act=None):
        super(FeedForwardLayer, self).__init__()
        # self.linear = nn.Linear(in_feat, out_feat, bias=bias)
        # self.W = nn.Parameter(t.zeros(args.latdim, args.latdim).cuda())
        self.W = nn.Parameter(t.eye(args.latdim, args.latdim).cuda(), requires_grad=False)
        self.bias = nn.Parameter(t.zeros( 1, args.latdim).cuda(), requires_grad=False)

        
        if act == 'identity' or act is None:
            self.act = None
        elif act == 'leaky':
            self.act = nn.LeakyReLU(negative_slope=args.leaky)
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'relu6':
            self.act = nn.ReLU6()
        else:
            raise Exception('Error')
    
    def forward(self, embeds):
        if self.act is None:
            # return self.linear(embeds)
            return  embeds @ self.W 
        # return (self.act(  embeds @ self.W + self.bias )) + embeds  #  default   v1
        return self.act(  embeds @ self.W + self.bias )  #  v2
    

class SGL(nn.Module):
    def __init__(self):
        super(SGL, self).__init__()

        self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
        self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

        self.edgeDropper = SpAdjDropEdge()

    def getEgoEmbeds(self, adj):
        uEmbeds, iEmbeds = self.forward(adj)
        return t.concat([uEmbeds, iEmbeds], axis=0)

    def forward(self, adj, iniEmbeds=None , keepRate=args.sglkeepRate):
        if iniEmbeds is None:
            iniEmbeds = t.concat([self.uEmbeds, self.iEmbeds], axis=0)

        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        mainEmbeds = sum(embedsLst) / len(embedsLst)

        if keepRate == 1.0 or self.training == False:
            return mainEmbeds[:args.user], mainEmbeds[args.user:]

        adjView1 = self.edgeDropper(adj, keepRate)
        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adjView1, embedsLst[-1])
            embedsLst.append(embeds)
        embedsView1 = sum(embedsLst)

        adjView2 = self.edgeDropper(adj, keepRate)
        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adjView2, embedsLst[-1])
            embedsLst.append(embeds)
        embedsView2 = sum(embedsLst)
        return mainEmbeds[:args.user], mainEmbeds[args.user:], embedsView1[:args.user], embedsView1[args.user:], embedsView2[:args.user], embedsView2[args.user:]		
     
    def cal_loss(self, batch_data,  tuned_emb=None , ori_adj=None, ts_pk_adj=None , mask=None, ts_drp_adj=None, drp_edges=None,  pk_edges=None):     
        # ancs, poss, negs = batch_data
        ancs, poss, negs = list(map(lambda x: x.long(), batch_data))
        ancs = ancs.long().cuda()
        poss = poss.long().cuda()
        negs = negs.long().cuda()
        usrEmbeds, itmEmbeds, usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2 = self.forward(ts_pk_adj, iniEmbeds=tuned_emb ,keepRate = args.sglkeepRate)
        ancEmbeds = usrEmbeds[ancs]
        posEmbeds = itmEmbeds[poss]
        negEmbeds = itmEmbeds[negs]

        clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, args.sgltemp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, args.sgltemp)) * args.sgl_ssl_reg

        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        # bprLoss = - (scoreDiff).sigmoid().log().sum()
        bprLoss = - ((scoreDiff).sigmoid() + 1e-8 ).log().mean()
        regLoss = calcRegLoss([self.uEmbeds[ancs], self.iEmbeds[poss], self.iEmbeds[negs]]) * args.reg
        # regLoss = calcRegLoss(self.model) * args.reg
        loss = bprLoss + regLoss + clLoss

        loss_dict = {'bpr_loss': bprLoss, 'reg_loss': regLoss}

        return loss, loss_dict

    def full_predict(self, usrs, trn_mask, adj):
        self.training = False
        usr_embeds, itm_embeds = self.forward(adj, keepRate=1.0)
        pck_usr_embeds = usr_embeds[usrs]
        full_preds = pck_usr_embeds @ itm_embeds.T
        full_preds = full_preds * (1 - trn_mask) - trn_mask * 1e8
        return full_preds
