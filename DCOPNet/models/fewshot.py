import numpy as np
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .encoder import Res101Encoder
from .attention import MultiHeadAttention
from .attention import MultiLayerPerceptron

##onion pool
def erode_mask(mask, kernel_size=3, iterations=1):
    """
    Erode a binary mask using max pooling on its inverse.

    Args:
    - mask (torch.Tensor): Binary mask tensor to erode.
    - kernel_size (int): Size of the max pooling kernel.
    - iterations (int): Number of erosion iterations.

    Returns:
    - torch.Tensor: Eroded binary mask.
    """
    for _ in range(iterations):
        # Invert the mask: 1s become 0s and 0s become 1s
        inverted_mask = 1 - mask
        # Apply max pooling to the inverted mask
        pooled_inverted_mask = F.max_pool2d(inverted_mask, kernel_size, stride=1, padding=kernel_size // 2)
        # Invert the result back
        mask = 1 - pooled_inverted_mask
    return mask

def generate_onion_layers(foreground_mask, num_layers):
    """
    Generate "onion" layers of foreground prototypes using PyTorch.

    Args:
    - foreground_mask (torch.Tensor): Binary mask of the foreground (1 for foreground, 0 for background) with shape (1, 64, 64).
    - num_layers (int): Number of layers (prototypes) to generate.

    Returns:
    - List of torch.Tensor: A list of binary masks, each representing a layer of the "onion".
    """
    layers = []
    current_mask = foreground_mask.clone()
    #layers.append(current_mask.clone())

    for i in range(num_layers):

        # Check if there's any foreground left
        if torch.sum(current_mask) == 0:
            break
        if i!= 0:
           # Erode the current mask to create the next inner layer
            current_mask = erode_mask(current_mask, kernel_size=2, iterations=1)
           # Add the current layer
            layers.append(current_mask.clone())

       # print(current_mask.shape)
    return layers

class ResidualPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualPoolingModule, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.conv1 = nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1)  # Update input channels here
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)  # Update input channels here
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.fc = nn.Linear(in_channels, out_channels)  # Update output size to match input size
        self.c = in_channels
    def forward(self, x):
        #x: 1,256,1,1   in:256 out:256
        device = x.device
        x_upsampled = self.upsample(x)
        x_max = self.max_pool(x_upsampled)
        x_avg = self.avg_pool(x_upsampled)
        x_concat = torch.cat((x_max, x_avg), dim=1)
        x_concat = x_concat.to(self.conv1.weight.device)  # Convert input tensor to the same device as weights
        #print("2", x_concat.shape)
        x_res = self.conv1(x_concat)
        x_res = self.relu(x_res)
        x_res = self.conv2(x_res)
        x_res = self.relu(x_res)
        x_res = x_res.view(x_res.size(0), -1) #将尺寸从 1x256x1x1 展平为 1x256。
        x_res = self.fc(x_res) #1x256 变为 1x256。
        x_res = x_res.view(-1, self.c, 1, 1)
        x_res = x_res.to(device)

        x_out = x_res + x
        return x_out

class ThresholdNet(nn.Module):
    def __init__(self):
        super(ThresholdNet, self).__init__()
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.fc = nn.Linear(256 * 16 * 16, 1)

    def forward(self, Fs, Fs1):
        x = torch.cat((Fs, Fs1), dim=1)  # Concatenate Fs and Fs1 along channel dimension
        #print(x.shape)
        x = self.conv1(x)  #  [1,512,32,32]->[1,256,32,32]
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#PTP module
class ParallelConvNet(nn.Module):
    def __init__(self):
        super(ParallelConvNet, self).__init__()
        self.conv1a = nn.Conv2d(1024, 256, kernel_size=3, padding=1)
        #ct 时self.conv1b 中的padding=0，mri padding=1
        self.conv1b = nn.Conv2d(1024, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.fc = nn.Linear(256 * 16 * 16, 1)

    def forward(self, Fs, Fs1):
        x1 = torch.cat((Fs, Fs1), dim=1)  # Concatenate Fs and Fs1 along channel dimension
        x2 = x1.clone()  # Create a copy for the second branch

        # Branch 1: Max pooling + Convolution
        x1 = self.conv1a(x1)
        x1 = nn.functional.relu(x1)
        x1 = nn.functional.max_pool2d(x1, 2)

        # Branch 2: Convolution
        x2 = self.conv1b(x2)
        x2 = nn.functional.relu(x2)
        x2 = nn.functional.avg_pool2d(x2, 2)

        # Concatenate features from both branches
        x = torch.cat((x1, x2), dim=1)

        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class FewShotSeg(nn.Module):

    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()

        # Encoder
        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)  # or "resnet101"
        self.device = torch.device('cuda')
        self.scaler = 20.0
        self.criterion = nn.NLLLoss()
        self.criterion_MSE = nn.MSELoss()
        self.fg_sampler = np.random.RandomState(1289)
        self.fg_num = 100  # number of foreground partitions
        self.MHA = MultiHeadAttention(n_head=3, d_model=512, d_k=512, d_v=512)
        self.MLP = MultiLayerPerceptron(dim=512, mlp_dim=1024)
        self.layer_norm = nn.LayerNorm(512)
        self.CrossAttention_gate = CrossAttention_gate(512, 64)
        self.SelfProAttention =  SelfProAttention(512)
        self.conv_fusion = nn.Conv2d(2* 512 + 1, 512, kernel_size=1)
        self.conv_fusionsupp = nn.Conv2d(2* 512 + 1, 512, kernel_size=1, padding=0, bias=False)
        self.thres2 = ParallelConvNet()

        self.high_avg_pool = nn.AdaptiveAvgPool1d(256)
    def negSim(self, fts, prototype):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """

        sim = - F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler

        return sim
    def getLPred(self, sim, thresh):
        pred = []

        for s, t in zip(sim, thresh):
            pred.append(1.0 - torch.sigmoid(0.5 * (s - t)))

        return torch.stack(pred, dim=1)  # N x Wa x H' x W'

    #generate prior mask
    def Newgenerate_prior(self, query_feat,fts_size,fore_supp_pro):
        ### query_feat [1,256,32,32], supp_feat [1,256,32,32]
        ###        s_y [1,256，256],   fts_size [32,32] fore_supp_pro
        bsize, _, sp_sz, _ = query_feat.size()[:]
        ### corr_query 1，1，32，32
        anom_s = self.negSim(query_feat, fore_supp_pro).unsqueeze(0)
        corr_query = self.getLPred(anom_s, self.thresh_pred)

        corr_query = F.interpolate(corr_query, size=(fts_size[0], fts_size[1]), mode='bilinear', align_corners=True)
        return corr_query

    def forward(self, supp_imgs, supp_mask, qry_imgs, qry_mask, train=False, t_loss_scaler=1, n_iters=20):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """
        n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.criterion_MSE = nn.MSELoss()
        self.n_queries = len(qry_imgs)
        self.thresh_pred = torch.Tensor([-10.5])
        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        self.iter = 3
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1
        n_queries = len(qry_imgs)
        batch_size_q = qry_imgs[0].shape[0]
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]



        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  # B x Wa x Sh x H x W

        # Dilate the mask
        kernel = np.ones((3, 3), np.uint8)
        supp_mask_ = supp_mask.cpu().numpy()[0][0][0]
        supp_dilated_mask = cv2.dilate(supp_mask_, kernel, iterations=1)  # (256, 256)
        supp_periphery_mask = supp_dilated_mask - supp_mask_
        supp_periphery_mask = np.reshape(supp_periphery_mask, (supp_bs, self.n_ways, self.n_shots,
                                                               np.shape(supp_periphery_mask)[0],
                                                               np.shape(supp_periphery_mask)[1]))
        supp_dilated_mask = np.reshape(supp_dilated_mask, (supp_bs, self.n_ways, self.n_shots,
                                                           np.shape(supp_dilated_mask)[0],
                                                           np.shape(supp_dilated_mask)[1]))
        supp_periphery_mask = torch.tensor(supp_periphery_mask).cuda()  # (1, 1, 1, 256, 256)  B x Wa x Sh x H x W
        supp_dilated_mask = torch.tensor(supp_dilated_mask).cuda()  # (1, 1, 1, 256, 256)  B x Wa x Sh x H x W

        # Extract features #
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        img_fts, tao = self.encoder(imgs_concat)
        fts_size = img_fts.shape[-2:]
        supp_fts = img_fts[:self.n_ways * self.n_shots * supp_bs].view(  # B x Wa x Sh x C x H' x W'
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts.shape[-2:])

        qry_fts = img_fts[self.n_ways * self.n_shots * supp_bs:].view(  # B x N x C x H' x W'
            qry_bs, self.n_queries, -1, *img_fts.shape[-2:])

##### DFCA module
        # Reshape for attention
        supp_fts_reshaped = supp_fts.view(-1, *supp_fts.shape[-3:])  # (Wa*Sh*B) x C x H' x W'
        qry_fts_reshaped = qry_fts.view(-1, *qry_fts.shape[-3:])  # (N*B) x C x H' x W'

        # Reshape back to original size
        supp_fts = supp_fts_reshaped.view(n_ways, self.n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = qry_fts_reshaped.view(n_queries, batch_size_q, -1, *fts_size)  # N x B x C x H' x W'
        fore_mask = supp_mask

        #step1 ###### Generate prior ######
        qry_fts1 = qry_fts.view(-1, qry_fts.shape[2], *fts_size)  # (N * B) x C x H' x W'
        supp_fts1 = supp_fts.view(batch_size, -1, *fts_size)  # B x C x H' x W'
        fore_mask1 = fore_mask[0][0]  # B x H' x W'

        fore_supp_pro = self.getFeatures(supp_fts1, fore_mask1)
       # corr_query_mask = self.Newgenerate_prior(qry_fts1, (64, 64), fore_supp_pro)
        fts_size = qry_fts[0][0].shape[-2:]
        corr_query_mask = self.Newgenerate_prior(qry_fts1,  (fts_size[0], fts_size[1]), fore_supp_pro)

        #step2 feature fusion

        fore_supp_pro = fore_supp_pro.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)  # 1,1,512,1,1
        fore_supp_pro = fore_supp_pro.expand_as(qry_fts)  # 1,1, 512, 64,64

        # Reshape corr_query_mask from (N * B) x 1 x H' x W' to N x B x 1 x H' x W'
        corr_query_mask = corr_query_mask.view(n_queries, batch_size_q, 1,  fts_size[0], fts_size[1])
        # Fusion prior and query features

        qry_fts = torch.cat([qry_fts, fore_supp_pro, corr_query_mask], dim=2)  # N x B x (C + 1) x H' x W'
        qry_fts = self.conv_fusion(qry_fts.view(-1, qry_fts.shape[2], *fts_size)).view(n_queries, batch_size_q, -1,
                                                                                       *fts_size)
        fore_mask1_pooled = F.adaptive_avg_pool2d(fore_mask1, (fts_size[0], fts_size[1]))

        #fts_size[0], fts_size[1]
        supp_fts = torch.cat([supp_fts_reshaped.view(-1, 1, qry_fts.shape[2], *fts_size), fore_supp_pro,
                              fore_mask1_pooled.unsqueeze(0).unsqueeze(0)], dim=2)  # N x B x (C + 1) x H' x W'
        supp_fts = self.conv_fusionsupp(supp_fts.view(-1, supp_fts.shape[2], *fts_size)).view(n_queries, batch_size_q,
                                                                                              -1,
                                                                                              *fts_size)
        supp_fts_reshaped = supp_fts.view(-1, *supp_fts.shape[2:])
        qry_fts_reshaped = qry_fts.view(-1, *qry_fts.shape[2:])


        #step3 Pass through CrossAttention
        supp_fts_out, qry_fts_out = self.CrossAttention_gate(supp_fts_reshaped, qry_fts_reshaped, fore_mask1_pooled,
                                                             corr_query_mask.squeeze(0).squeeze(0))
        #print(supp_fts_reshaped.shape, supp_fts_out.shape)
        thre_sup = self.thres2(supp_fts_reshaped, supp_fts_out)
        thre_qry = self.thres2(qry_fts_reshaped, qry_fts_out)

        supp_fts = supp_fts_out.unsqueeze(0).unsqueeze(0)

###### # Get adaptive threshold
        self.t = thre_qry   # t for query features
        self.thresh_pred = [self.t for _ in range(self.n_ways)]
        self.t_ = thre_sup  # t for support features
        self.thresh_pred_ = [self.t_ for _ in range(self.n_ways)]


        # Compute loss #
        periphery_loss = torch.zeros(1).to(self.device)
        align_loss = torch.zeros(1).to(self.device)
        mse_loss = torch.zeros(1).to(self.device)
        qry_loss = torch.zeros(1).to(self.device)
        outputs = []
        qryputs = []
        for epi in range(supp_bs):
            # Partition the foreground object into N parts, the coarse support prototypes
            fg_partition_prototypes = [[self.compute_multiple_prototypes(
                self.fg_num, supp_fts[[epi], way, shot], supp_mask[[epi], way, shot], self.fg_sampler)
                for shot in range(self.n_shots)] for way in range(self.n_ways)]

            # calculate coarse query prototype
            supp_fts_ = [[self.getFeatures(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot])
                          for shot in range(self.n_shots)] for way in range(self.n_ways)]

            fg_prototypes = self.getPrototype(supp_fts_)  # the coarse foreground

            # Dilated region prototypes #
            supp_fts_dilated = [[self.getFeatures(supp_fts[[epi], way, shot], supp_dilated_mask[[epi], way, shot])
                                  for shot in range(self.n_shots)] for way in range(self.n_ways)]
            fg_prototypes_dilated = self.getPrototype(supp_fts_dilated)

            # Segment periphery region with support images
            supp_pred_object = torch.stack([self.getPred(supp_fts[epi][way], fg_prototypes[way], self.thresh_pred_[way])
                             for way in range(self.n_ways)], dim=1)   # N x Wa x H' x W'
            supp_pred_object = F.interpolate(supp_pred_object, size=img_size, mode='bilinear', align_corners=True)

            supp_pred_dilated = torch.stack([self.getPred(supp_fts[epi][way], fg_prototypes_dilated[way], self.thresh_pred_[way])
                             for way in range(self.n_ways)], dim=1)   # N x Wa x H' x W'
            supp_pred_dilated = F.interpolate(supp_pred_dilated, size=img_size, mode='bilinear', align_corners=True)

            # Prediction of periphery region
            pred_periphery = supp_pred_dilated - supp_pred_object
            pred_periphery = torch.cat((1.0 - pred_periphery, pred_periphery), dim=1)
            # pred_periphery: (1, 2, 256, 256)  B x C x H x W
            label_periphery = torch.full_like(supp_periphery_mask[epi][0][0], 255, device=supp_periphery_mask.device)
            label_periphery[supp_periphery_mask[epi][0][0] == 1] = 1
            label_periphery[supp_periphery_mask[epi][0][0] == 0] = 0
            # label_periphery: (256, 256)  H x W

            # Compute periphery loss
            eps_ = torch.finfo(torch.float32).eps
            log_prob_ = torch.log(torch.clamp(pred_periphery, eps_, 1 - eps_))
            periphery_loss += self.criterion(log_prob_, label_periphery[None, ...].long()) / self.n_shots / self.n_ways

            qry_pred = torch.stack(
                [self.getPred(qry_fts[epi], fg_prototypes[way], self.thresh_pred[way])
                 for way in range(self.n_ways)], dim=1)  # N x Wa x H' x W'

            qry_prototype_coarse = self.getFeatures(qry_fts[epi], qry_pred[epi])

            # # The first BATE block
            for i in range(self.iter):
                fg_partition_prototypes = [[self.BATE(fg_partition_prototypes[way][shot][epi], qry_prototype_coarse)
                                            for shot in range(self.n_shots)] for way in range(self.n_ways)]


                supp_proto = [[torch.mean(fg_partition_prototypes[way][shot], dim=1) + fg_prototypes[way] for shot in range(self.n_shots)]
                              for way in range(self.n_ways)]

                # CQPC module
                qry_pred_coarse = torch.stack(
                    [self.getPred(qry_fts[epi], supp_proto[way][epi], self.thresh_pred[way])
                     for way in range(self.n_ways)], dim=1)

                qry_prototype_coarse = self.getFeatures(qry_fts[epi], qry_pred_coarse[epi])

            qry_pred = torch.stack(
                [self.getPred(qry_fts[epi], supp_proto[way][epi], self.thresh_pred[way])
                 for way in range(self.n_ways)], dim=1)  # N x Wa x H' x W'

            Secqry_pred = self.NewupdatePrototype(qry_fts[epi], qry_pred.squeeze(0), img_size)
            qryputs.append(Secqry_pred)

            # Combine predictions of different feature maps #
            qry_pred_up = F.interpolate(qry_pred, size=img_size, mode='bilinear', align_corners=True)

            preds = torch.cat((1.0 - qry_pred_up, qry_pred_up), dim=1)

            outputs.append(preds)

            if train:
                align_loss_epi = self.alignLoss(supp_fts[epi], qry_fts[epi], preds, supp_mask[epi])
                align_loss += align_loss_epi
            if train:
                proto_mse_loss_epi = self.proto_mse(qry_fts[epi], preds, supp_mask[epi], fg_prototypes)
                mse_loss += proto_mse_loss_epi
            if train:
                qry_fts_ = [[self.getFeatures(qry_fts[epi], qry_mask)]]
                qry_prototypes = self.getPrototype(qry_fts_)
                qry_pred = self.getPred(qry_fts[epi], qry_prototypes[epi], self.thresh_pred[epi])

                qry_pred = F.interpolate(qry_pred[None, ...], size=img_size, mode='bilinear', align_corners=True)
                preds = torch.cat((1.0 - qry_pred, qry_pred), dim=1)

                qry_label = torch.full_like(qry_mask[epi], 255, device=qry_mask.device)
                qry_label[qry_mask[epi] == 1] = 1
                qry_label[qry_mask[epi] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(preds, eps, 1 - eps))
                qry_loss += self.criterion(log_prob, qry_label[None, ...].long()) / self.n_shots / self.n_ways

        output = torch.stack(outputs, dim=1)
        output = output.view(-1, *output.shape[2:])

        qryputs = torch.stack(qryputs, dim=1)  # N x B x (1 + Wa) x H x W
        qryputs = qryputs.view(-1, *qryputs.shape[2:])
        return output, periphery_loss / supp_bs, align_loss / supp_bs, mse_loss / supp_bs, qry_loss / supp_bs,qryputs

    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))

        return pred

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')

        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        return masked_fts

    def getFeatures_fg(self, fts, mask):
        """
        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """

        fts_ = fts.squeeze(0).permute(1, 2, 0)

        fts_ = fts_.view(fts_.size()[0] * fts_.size()[1], fts_.size()[2])
        mask_ = F.interpolate(mask.unsqueeze(0), size=fts.shape[-2:], mode='bilinear')
        mask_ = mask_.view(-1)

        l = math.ceil(mask_.sum())
        c = torch.argsort(mask_, descending=True, dim=0)
        fg = c[:l]

        fts_fg = fts_[fg]

        return fts_fg

    def NewupdatePrototype(self, qry, pred_mask,img_size): #QSR loss

        qry_protos = self.getNewProtos(qry, pred_mask)
        anom_s = self.negSim(qry, qry_protos)
        pred = self.getLPred([anom_s], self.thresh_pred)  # N x Wa x H' x W'  1,1,32,32

        #插值到原形状
        pred_ups = F.interpolate(pred, size=img_size, mode='bilinear', align_corners=True)  # 1,1,256,256
        pred_ups = torch.cat((1.0 - pred_ups, pred_ups), dim=1)

        return pred_ups
    def getPrototype(self, fg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  ## concat all fg_fts

        return fg_prototypes
    def getNewProtos(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        residual_pooling_module = ResidualPoolingModule(512, 512)

        # fts  1, 256, 32, 32    ,1，256，256，256
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear') #插值到相同大小
        # fts‘ 1，256，256，256      mask 1，256，256

        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
                     / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)  # 1 x C

        #ResidualDeconvModule
        mfts = masked_fts.unsqueeze(2).unsqueeze(3) # 1,256 -> 1,256,1,1
        #print("1",mfts.shape)
        output_feature = residual_pooling_module(mfts)
        output_feature = output_feature.squeeze(2).squeeze(2) #1,256
        return output_feature

    def compute_multiple_prototypes(self, fg_num, sup_fts, sup_fg, sampler):
        """

        Parameters
        ----------
        fg_num: int
            Foreground partition numbers
        sup_fts: torch.Tensor
             [B, C, h, w], float32
        sup_fg: torch. Tensor
             [B, h, w], float32 (0,1)
        sampler: np.random.RandomState

        Returns
        -------
        fg_proto: torch.Tensor
            [B, k, C], where k is the number of foreground proxies

        """

        fg_protos1 = []
        num_layers = 4 # Number of layers to generate
        layers = generate_onion_layers(sup_fg.unsqueeze(0).float(), num_layers)
        ps = self.getFeatures(sup_fts, sup_fg)  #sup_fts, sup_fg
        fg_protos1.append(ps)  # n,1,c
        for i in range(len(layers)):
            ps = self.getFeatures(sup_fts, layers[i])  #1*c
            fg_protos1.append(ps.squeeze(0))  #n,1,c


        fg_protos1 = torch.stack(fg_protos1, dim=1)  #  fg_protos1[0].shape [4,512]     torch.Size([1,3, 512])
        enhanced_prototypes = self.SelfProAttention(fg_protos1[0])  #enhanced_prototypes [4,512]

        return   [enhanced_prototypes]

    def BATE(self, fg_prototypes, qry_prototype_coarse):

        # S&W module
        A = torch.mm(fg_prototypes, qry_prototype_coarse.t())
        kc = ((A.min() + A.mean()) / 2).floor()

        if A is not None:
            S = torch.zeros(A.size(), dtype=torch.float).cuda()
            S[A < kc] = -10000.0

        A = torch.softmax((A + S), dim=0)
        # fg_prototypes = A * fg_prototypes
        A = torch.mm(A, qry_prototype_coarse)
        A = self.layer_norm(A + fg_prototypes)

        # rest Transformer operation
        T = self.MHA(A.unsqueeze(0), A.unsqueeze(0), A.unsqueeze(0))
        T = self.MLP(T)


        return T

    def alignLoss(self, supp_fts, qry_fts, pred, fore_mask):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                qry_fts_ = [self.getFeatures(qry_fts, pred_mask[way + 1])]
                fg_prototypes = self.getPrototype([qry_fts_])

                # Get predictions
                supp_pred = self.getPred(supp_fts[way, [shot]], fg_prototypes[way], self.thresh_pred[way])  # N x Wa x H' x W'
                supp_pred = F.interpolate(supp_pred[None, ...], size=fore_mask.shape[-2:], mode='bilinear',
                                           align_corners=True)


                # Combine predictions of different feature maps
                preds = supp_pred
                pred_ups = torch.cat((1.0 - preds, preds), dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss


    def proto_mse(self, qry_fts, pred, fore_mask, supp_prototypes):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss_sim = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                qry_fts_ = [[self.getFeatures(qry_fts, pred_mask[way + 1])]]

                fg_prototypes = self.getPrototype(qry_fts_)

                fg_prototypes_ = torch.sum(torch.stack(fg_prototypes, dim=0), dim=0)
                supp_prototypes_ = torch.sum(torch.stack(supp_prototypes, dim=0), dim=0)


                # Compute the MSE loss

                loss_sim += self.criterion_MSE(fg_prototypes_, supp_prototypes_)

        return loss_sim


class SelfProAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SelfProAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):  #x.shape 4,512

        # x shape: (n, feature_dim), where n is the number of prototypes 如【3，512】
        Q = self.query(x)  # Query
        K = self.key(x)  # Key
        V = self.value(x)  # Value

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention weights to values
        attention_output = torch.matmul(attention_weights, V)

        return attention_output

class CrossAttention_gate(nn.Module):
    def __init__(self, dim,fts):
        super(CrossAttention_gate, self).__init__()
        self.query = nn.Conv2d(dim, dim // 8, 1)
        self.key = nn.Conv2d(dim, dim // 8, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm([1,512,fts, fts])
        #self.threshold = nn.Parameter(torch.tensor(0.0))  # Add threshold parameter

    def forward(self, x, y, s_m, q_m):
        B, C, H, W = x.shape
        scale = (C // 8) ** -0.5
        #阈值化q_m
        q_m = torch.where(q_m > 0.8, torch.tensor(1.0), torch.tensor(0.0))
        # Obtain foreground regions
        x_foreground = x * s_m.view(1, 1, H, W)
        y_foreground = y * q_m.view(1, 1, H, W)

        qx = self.query(x_foreground).view(B, -1, H * W).permute(0, 2, 1) * scale
        ky = self.key(y_foreground).view(B, -1, H * W)
        vy = self.value(y_foreground).view(B, -1, H * W)

        attn1 = torch.bmm(qx, ky)
        # Apply threshold to adjust attention weights
        threshold_value = (torch.max(attn1) + torch.mean(attn1)) / 2
        attn1 = torch.where(attn1 < threshold_value, torch.tensor(-999.0, device=x.device), attn1)
        attn1 = self.softmax(attn1)

        outx = torch.bmm(vy, attn1.permute(0, 2, 1)).view(B, C, H, W)
        outx = self.mlp(outx.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        #print(outx.shape)
        outx = self.norm(outx)
        # Obtain background regions
        x_background = x * (1 - s_m).view(1, 1, H, W)
        outx = outx * s_m.view(1, 1, H, W) + x_background


        qy = self.query(y_foreground).view(B, -1, H * W).permute(0, 2, 1) * scale
        kx = self.key(x).view(B, -1, H * W)
        vx = self.value(x).view(B, -1, H * W)

        attn2 = torch.bmm(qy, kx)

        # Apply threshold to adjust attention weights
        threshold_value = (torch.min(attn2) + torch.mean(attn2)) / 2
        attn2 = torch.where(attn2 < threshold_value, torch.tensor(-999.0, device=x.device), attn2)
        attn2 = self.softmax(attn2)

        outy = torch.bmm(vx, attn2.permute(0, 2, 1)).view(B, C, H, W)
        outy = self.mlp(outy.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        outy = self.norm(outy)

        # Obtain background regions
        y_background = y * (1 - q_m).view(1, 1, H, W)
        #y_background = self.self_attention(y_background)
        outy = outy * q_m.view(1, 1, H, W) + y_background
        return outx, outy













