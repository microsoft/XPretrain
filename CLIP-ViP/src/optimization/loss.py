import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class TripletContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, cfg):
        super(TripletContrastiveLoss, self).__init__()
        self.margin = cfg.margin
        if cfg.measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = cfg.max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class NCEContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, cfg):
        super(NCEContrastiveLoss, self).__init__()
        self.temp = cfg.temp

    def forward(self, vis_feat, text_feat):

        t2v = torch.matmul(vis_feat, text_feat.permute(1, 0)) / self.temp  # temperature
        v2t = t2v.permute(1, 0)
        t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
        v2t_label = t2v_label
        loss = (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label)).mean()
        return loss


class HardNegLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, cfg):
        super(HardNegLoss, self).__init__()
        self.hard_negative_num = cfg.hard_negative_num

    def forward(self, vis_feat, text_feat):
        sim_matrix = torch.matmul(text_feat, vis_feat.permute(1, 0))  # temperature
        bsz = sim_matrix.shape[0]
        retrieval_mask = torch.eye(bsz, dtype=torch.long).to(device=sim_matrix.device)
        hard_neg_t2v = torch.topk(sim_matrix-10000*retrieval_mask, self.hard_negative_num, dim=1)[0]
        hard_neg_v2t = torch.topk(sim_matrix.transpose(0, 1)-10000*retrieval_mask, self.hard_negative_num, dim=1)[0]
        sample_t2v = torch.cat([sim_matrix.diag().view(-1, 1), hard_neg_t2v], -1)
        sample_v2t = torch.cat([sim_matrix.diag().view(-1, 1), hard_neg_v2t], -1)
        retrieval_label = torch.zeros(bsz, dtype=torch.long).to(device=sim_matrix.device)
        loss = (F.cross_entropy(sample_t2v, retrieval_label) + F.cross_entropy(sample_v2t, retrieval_label)).mean()
        return loss



class MILNCEContrastiveLoss(nn.Module):
    def __init__(self,cfg):
        super(MILNCEContrastiveLoss, self).__init__()
        self.temp = cfg.temp

    def forward(self, video_embd, text_embd):
        x = torch.matmul(video_embd, text_embd.t()) / self.temp

        x = x.view(video_embd.shape[0], video_embd.shape[0], -1)
        nominator = x * torch.eye(x.shape[0])[:,:,None].to(x.device)
        nominator = nominator.sum(dim=1)
        nominator = torch.logsumexp(nominator, dim=1)
        mask = (1-torch.eye(x.shape[0])[:,:,None].to(x.device).repeat(1,1,x.shape[-1]))
        denominator = torch.cat((x[mask>0].reshape(x.shape[0], x.shape[1]-1, x.shape[2]), x.permute(1,0,2)), dim=1).view(x.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)
        return torch.mean(denominator - nominator)

class NCELearnableTempLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, cfg):
        super(NCELearnableTempLoss, self).__init__()

    def forward(self, vis_feat, text_feat, temp):
        logit_scale = temp.exp()
        t2v = torch.matmul(vis_feat, text_feat.permute(1, 0)) * logit_scale  # temperature
        v2t = t2v.permute(1, 0)
        t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
        v2t_label = t2v_label
        loss = (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label)).mean()
        return loss

class VidImgNCELearnableTempLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, cfg):
        super(VidImgNCELearnableTempLoss, self).__init__()

    def forward(self, vis_feat, text_feat, img_feat, cap_feat, temp):
        vis_feat = torch.cat([vis_feat, img_feat], dim=0)
        text_feat = torch.cat([text_feat, cap_feat], dim=0)
        logit_scale = temp.exp()
        t2v = torch.matmul(vis_feat, text_feat.permute(1, 0)) * logit_scale  # temperature
        v2t = t2v.permute(1, 0)
        t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
        v2t_label = t2v_label
        loss = (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label)).mean()
        return loss

class VidImgDivideNCELearnableTempLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, cfg):
        super(VidImgDivideNCELearnableTempLoss, self).__init__()

    def forward(self, vis_feat, text_feat, img_feat, cap_feat, temp):
        logit_scale = temp.exp()
        t2v = torch.matmul(vis_feat, text_feat.permute(1, 0)) * logit_scale  # temperature
        v2t = t2v.permute(1, 0)
        t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
        v2t_label = t2v_label

        t2v_2 = torch.matmul(img_feat, cap_feat.permute(1, 0)) * logit_scale  # temperature
        v2t_2 = t2v_2.permute(1, 0)
        t2v_label_2 = torch.arange(t2v_2.shape[0], device=t2v_2.device)
        v2t_label_2 = t2v_label_2
        loss = (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label) +\
            F.cross_entropy(t2v_2, t2v_label_2) + F.cross_entropy(v2t_2, v2t_label_2)).mean()
        return loss

class NCELearnableTempDSLLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, cfg):
        super(NCELearnableTempDSLLoss, self).__init__()

    def forward(self, vis_feat, text_feat, temp):
        logit_scale = temp.exp()
        t2v = torch.matmul(vis_feat, text_feat.permute(1, 0)) * logit_scale  # temperature
        v2t = t2v.permute(1, 0)
        t2v = t2v * F.softmax(t2v, dim=0)
        v2t = v2t * F.softmax(v2t, dim=0)
        t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
        v2t_label = t2v_label
        loss = (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label)).mean()
        return loss

class NCELearnableTempLoss_vs_vc(nn.Module):
    """
    Compute contrastive loss: video-sub + video-cap
    """

    def __init__(self, cfg):
        super(NCELearnableTempLoss_vs_vc, self).__init__()

    def forward(self, vis_feat, text_feat, img_feat, cap_feat, temp):
        logit_scale = temp.exp()
        t2v = torch.matmul(vis_feat, text_feat.permute(1, 0)) * logit_scale  # temperature
        v2t = t2v.permute(1, 0)
        t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
        v2t_label = t2v_label

        t2v_2 = torch.matmul(vis_feat, cap_feat.permute(1, 0)) * logit_scale  # temperature
        v2t_2 = t2v_2.permute(1, 0)
        t2v_label_2 = torch.arange(t2v_2.shape[0], device=t2v_2.device)
        v2t_label_2 = t2v_label_2
        loss = (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label) + \
            F.cross_entropy(t2v_2, t2v_label_2) + F.cross_entropy(v2t_2, v2t_label_2)).mean()
        return loss

class NCELearnableTempLoss_vs_vc_fc(nn.Module):
    """
    Compute contrastive loss: video-sub + video-cap
    """

    def __init__(self, cfg):
        super(NCELearnableTempLoss_vs_vc_fc, self).__init__()

    def forward(self, vis_feat, text_feat, img_feat, cap_feat, temp):
        logit_scale = temp.exp()
        t2v = torch.matmul(vis_feat, text_feat.permute(1, 0)) * logit_scale  # temperature
        v2t = t2v.permute(1, 0)
        t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
        v2t_label = t2v_label

        t2v_2 = torch.matmul(vis_feat, cap_feat.permute(1, 0)) * logit_scale  # temperature
        v2t_2 = t2v_2.permute(1, 0)
        t2v_label_2 = torch.arange(t2v_2.shape[0], device=t2v_2.device)
        v2t_label_2 = t2v_label_2

        t2v_3 = torch.matmul(img_feat, cap_feat.permute(1, 0)) * logit_scale  # temperature
        v2t_3 = t2v_3.permute(1, 0)
        t2v_label_3 = torch.arange(t2v_3.shape[0], device=t2v_3.device)
        v2t_label_3 = t2v_label_3
        loss = (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label) + \
            F.cross_entropy(t2v_2, t2v_label_2) + F.cross_entropy(v2t_2, v2t_label_2) +\
            F.cross_entropy(t2v_3, t2v_label_3) + F.cross_entropy(v2t_3, v2t_label_3)).mean()
        return loss

class NCELearnableTempLoss_vsc(nn.Module):
    """
    Compute contrastive loss: video-(sub,cap)
    """

    def __init__(self, cfg):
        super(NCELearnableTempLoss_vsc, self).__init__()

    def forward(self, vis_feat, text_feat, img_feat, cap_feat, temp):
        assert text_feat.shape[0] == cap_feat.shape[0]
        logit_scale = temp.exp()
        v2t = torch.matmul(vis_feat, text_feat.permute(1, 0)) * logit_scale  # temperature
        t2v = v2t.permute(1, 0)
        t2v_label = torch.arange(v2t.shape[0], device=v2t.device)

        v2t_2 = torch.matmul(vis_feat, cap_feat.permute(1, 0)) * logit_scale  # temperature
        t2v_2 = v2t_2.permute(1, 0)
        t2v_label_2 = torch.arange(v2t_2.shape[0], device=v2t_2.device)

        diag = torch.eye(v2t.shape[0], dtype=torch.bool).to(v2t.device)
        v2t_pos = v2t[diag].reshape(v2t.shape[0], 1)
        v2t_neg = v2t[~diag].reshape(v2t.shape[0], -1)
        v2t_pos_2 = v2t_2[diag].reshape(v2t_2.shape[0], 1)
        v2t_neg_2 = v2t_2[~diag].reshape(v2t_2.shape[0], -1)
        v2t = torch.cat([v2t_pos, v2t_neg, v2t_neg_2], dim=1)
        v2t_2 = torch.cat([v2t_pos_2, v2t_neg, v2t_neg_2], dim=1)
        v2t_label = torch.zeros(v2t.shape[0], dtype=torch.long).to(v2t.device)

        loss = (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(t2v_2, t2v_label_2) + \
            F.cross_entropy(v2t, v2t_label) + F.cross_entropy(v2t_2, v2t_label)).mean()
        return loss

class NCELearnableTempLoss_vsc_fc(nn.Module):
    """
    Compute contrastive loss: video-(sub,cap)
    """

    def __init__(self, cfg):
        super(NCELearnableTempLoss_vsc_fc, self).__init__()

    def forward(self, vis_feat, text_feat, img_feat, cap_feat, temp):
        assert text_feat.shape[0] == cap_feat.shape[0]
        logit_scale = temp.exp()
        v2t = torch.matmul(vis_feat, text_feat.permute(1, 0)) * logit_scale  # temperature
        t2v = v2t.permute(1, 0)
        t2v_label = torch.arange(v2t.shape[0], device=v2t.device)

        v2t_2 = torch.matmul(vis_feat, cap_feat.permute(1, 0)) * logit_scale  # temperature
        t2v_2 = v2t_2.permute(1, 0)
        t2v_label_2 = torch.arange(v2t_2.shape[0], device=v2t_2.device)

        diag = torch.eye(v2t.shape[0], dtype=torch.bool).to(v2t.device)
        v2t_pos = v2t[diag].reshape(v2t.shape[0], 1)
        v2t_neg = v2t[~diag].reshape(v2t.shape[0], -1)
        v2t_pos_2 = v2t_2[diag].reshape(v2t_2.shape[0], 1)
        v2t_neg_2 = v2t_2[~diag].reshape(v2t_2.shape[0], -1)
        v2t = torch.cat([v2t_pos, v2t_neg, v2t_neg_2], dim=1)
        v2t_2 = torch.cat([v2t_pos_2, v2t_neg, v2t_neg_2], dim=1)
        v2t_label = torch.zeros(v2t.shape[0], dtype=torch.long).to(v2t.device)

        v2t_3 = torch.matmul(img_feat, cap_feat.permute(1, 0)) * logit_scale  # temperature
        t2v_3 = v2t_3.permute(1, 0)
        t2v_label_3 = torch.arange(t2v_3.shape[0], device=t2v_3.device)
        v2t_label_3 = t2v_label_3

        loss = (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(t2v_2, t2v_label_2) + \
            F.cross_entropy(v2t, v2t_label) + F.cross_entropy(v2t_2, v2t_label) + \
            F.cross_entropy(t2v_3, t2v_label_3) + F.cross_entropy(v2t_3, v2t_label_3)).mean()
        return loss

def build_loss_func(cfg):
    loss_func = globals()[cfg.loss_name](cfg)
    return loss_func

if __name__ == '__main__':
    from easydict import EasyDict as edict
    cfg = edict({'loss_name':'MILNCELoss', 'temp':0.05})
    print(cfg.loss_name)
    loss_func = build_loss_func(cfg)
    print(loss_func.temp)
    video_embd = torch.randn(64,1024)
    text_embd = torch.randn(1280,1024)
    print(loss_func(video_embd, text_embd))


