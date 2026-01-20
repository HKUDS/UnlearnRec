import torch as t
import torch.nn.functional as F

def cal_bpr(anc_embeds, pos_embeds, neg_embeds):
	pos_preds = (anc_embeds * pos_embeds).sum(-1)
	neg_preds = (anc_embeds * neg_embeds).sum(-1)
	# print('loss', (pos_preds - neg_preds).sigmoid().log())
	# print('sigmoid', (pos_preds - neg_preds).sigmoid())
	return -((pos_preds - neg_preds).sigmoid() + 1e-10).log().mean()

def _crr_neg(embeds1, embeds2, temp):
	embeds1 = F.normalize(embeds1)
	embeds2 = F.normalize(embeds2)
	tem = embeds1 @ embeds2.T
	return t.log(t.exp((tem) / temp).sum(-1) + 1e-10).mean()

def cal_crr(usr_embeds, itm_embeds, ancs, poss, temp, inbatch=False):
	pos_term = 0#-(((usr_embeds[ancs] * itm_embeds[poss]).sum(-1))).mean()
	# pos_term = -(usr_embeds[ancs] * itm_embeds[poss]).sum(-1).mean()
	if not inbatch:
		neg_term = _crr_neg(usr_embeds[ancs], usr_embeds, temp) + _crr_neg(itm_embeds[poss], itm_embeds, temp) + _crr_neg(usr_embeds[ancs], itm_embeds, temp)
	else:
		neg_term = _crr_neg(usr_embeds[ancs], usr_embeds[ancs], temp) + _crr_neg(itm_embeds[poss], itm_embeds[poss], temp) + _crr_neg(usr_embeds[ancs], itm_embeds[poss], temp)
	return pos_term + neg_term

def cal_reg(model):
	ret = 0
	for W in model.parameters():
		ret += W.norm(2).square()
	return ret

def cal_neg_aug_v1(drp_uEmbeds, drp_iEmbeds):
	preds2unlearn = (drp_uEmbeds * drp_iEmbeds).sum(-1)
	return -( ( -preds2unlearn ).sigmoid() + 1e-10).log().mean()


def cal_neg_aug_v2(drp_uEmbeds, drp_iEmbeds):
	preds2unlearn = (drp_uEmbeds * drp_iEmbeds).sum(-1)
	# return -( ( -preds2unlearn ).sigmoid() + 1e-10).log().mean()
	return preds2unlearn.mean()

def cal_l2_distance(embeds1, embeds2):
	return (embeds1 - embeds2).norm(2).square().mean()


def innerProduct(usrEmbeds, itmEmbeds):
	return t.sum(usrEmbeds * itmEmbeds, dim=-1)

def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
	return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)
	
def cal_positive_pred_align(srcUEmbs, tarUEmbs, srcIEmbs , tarIEmbs, align ):
	srcpred = innerProduct(srcUEmbs, srcIEmbs)
	tarpred = innerProduct(tarUEmbs, tarIEmbs)
	loss = align(srcpred, tarpred)
	return loss

def cal_positive_pred_align_v2(srcUEmbs, tarUEmbs, srcIEmbs , tarIEmbs, align, temp=1. ):
	srcpred = innerProduct(srcUEmbs, srcIEmbs) / temp
	tarpred = innerProduct(tarUEmbs, tarIEmbs) / temp

	src_deno = srcUEmbs @ srcIEmbs.T / temp + 1e-8
	tar_deno = tarUEmbs @ tarIEmbs.T / temp + 1e-8

	src_dist = t.log(t.exp(srcpred) / (t.exp(src_deno).sum(-1) + 1e-6) + 1e-6 )
	tar_dist = t.log(t.exp(tarpred) / (t.exp(tar_deno).sum(-1) + 1e-6 ) + 1e-6 )

	loss = align(src_dist, tar_dist)

	return loss	


def cal_positive_pred_align_v3(srcUEmbs, tarUEmbs, srcIEmbs , tarIEmbs, align,  temp=0.1, activate = t.nn.Softplus(beta=1, threshold=20) ):
	srcpred = innerProduct(srcUEmbs, srcIEmbs) / temp
	tarpred = innerProduct(tarUEmbs, tarIEmbs) / temp

	src_deno = srcUEmbs @ srcIEmbs.T / temp + 1e-8
	tar_deno = tarUEmbs @ tarIEmbs.T / temp + 1e-8

	# print("####################src_deno, tar_deno############################")
	# print(src_deno)
	# print(tar_deno)	


	# src_deno = (t.exp(src_deno).sum(-1) + 1e-6)
	# tar_deno = (t.exp(tar_deno).sum(-1) + 1e-6 )

	
	src_dist = t.log(activate(srcpred) / (activate(src_deno).sum(-1) + 1e-6) + 1e-6 )
	tar_dist = t.log(activate(tarpred) / (activate(tar_deno).sum(-1) + 1e-6 ) + 1e-6 )


	# src_deno = (t.sigmoid(src_deno).sum(-1) + 1e-6)
	# tar_deno = (t.sigmoid(tar_deno).sum(-1) + 1e-6 )

	# src_dist = t.log(t.sigmoid(srcpred) / src_deno + 1e-6 )
	# tar_dist = t.log(t.sigmoid(tarpred) / tar_deno + 1e-6)

	# src_dist = t.log(t.exp(srcpred - src_deno)   + 1e-8 )
	# tar_dist = t.log(t.exp(tarpred - tar_deno)   + 1e-8 )
	# print("####################src_dist############################")
	# print(srcpred)
	# print(src_deno)
	# print(src_dist)
	# print("####################tar############################")
	# print(tarpred)
	# print(tar_deno)
	# print(tar_dist)

	loss = align(src_dist, tar_dist)

	# print("###############cal_positive_pred_align_v2####################")
	# print(loss)
	# loss = align(src_deno, tar_deno)
	return loss	










def calcRegLoss(params=None, model=None):
	ret = 0
	if params is not None:
		for W in params:
			ret += W.norm(2).square()
	if model is not None:
		for W in model.parameters():
			ret += W.norm(2).square()
	# ret += (model.usrStruct + model.itmStruct)
	return ret

def SimGCL_calcRegLoss(pck_usr_embeds, pck_itm_embeds):
	return pck_usr_embeds.norm(2) / pck_usr_embeds.shape[0] + pck_itm_embeds.norm(2) / pck_itm_embeds.shape[0]

def SimGCL_calcRegLoss_v2(pck_usr_embeds, pck_itm_embeds):
	return pck_usr_embeds.norm(2).square() / pck_usr_embeds.shape[0] + pck_itm_embeds.norm(2).square() / pck_itm_embeds.shape[0]


def SimGCL_calcRegLoss_v3(pck_usr_embeds, pck_itm_embeds):
	return pck_usr_embeds.norm(2).square() + pck_itm_embeds.norm(2).square()




def infoNCE(embeds1, embeds2, nodes, temp):
	embeds1 = F.normalize(embeds1 + 1e-8, p=2)
	embeds2 = F.normalize(embeds2 + 1e-8, p=2)
	pckEmbeds1 = embeds1[nodes]
	pckEmbeds2 = embeds2[nodes]
	nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
	deno = t.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1) + 1e-8
	return (-t.log(nume / deno)).mean()

def KLDiverge(tpreds, spreds, distillTemp=0.1):
	tpreds = (tpreds / distillTemp).sigmoid()
	spreds = (spreds / distillTemp).sigmoid()
	return -(tpreds * (spreds + 1e-8).log() + (1 - tpreds) * (1 - spreds + 1e-8).log()).mean()

def pointKLDiverge(tpreds, spreds):
	return -(tpreds * spreds.log()).mean()

def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()

def contrast(embeds1, embeds2, nodes, temp=10):
	uniqNodes = t.unique(nodes)
	pckEmbeds1 = F.normalize(embeds1[uniqNodes], p=2)
	pckEmbeds2 = F.normalize(embeds2[uniqNodes], p=2)
	posScores = (pckEmbeds1 * pckEmbeds2).sum(-1) / temp
	negScores = (pckEmbeds1 @ pckEmbeds2.T) / temp
	clLoss = -(t.log(t.exp(posScores) / t.exp(negScores).sum(-1))).mean()
	return clLoss

def contrastLoss(embeds1, embeds2, nodes, temp=10):
	embeds1 = F.normalize(embeds1 + 1e-8, p=2)
	embeds2 = F.normalize(embeds2 + 1e-8, p=2)
	pckEmbeds1 = embeds1[nodes]
	pckEmbeds2 = embeds2[nodes]
	nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
	deno = t.exp(pckEmbeds1 @ pckEmbeds2.T / temp).sum(-1) + 1e-8
	return -t.log(nume / deno).mean()