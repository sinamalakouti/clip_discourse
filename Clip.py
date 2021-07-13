import torch
import torch.nn as nn
import clip
import torch.distributed as dist


class ProjectionModel(torch.nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(ProjectionModel, self).__init__()

        self.logit_fc = torch.nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.BatchNorm1d(hid_dim, eps=1e-12),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim // 2)
        )

    def forward(self, X):
        return nn.functional.normalize(self.logit_fc(X), dim=1, p=2)


class PrototypeModel(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PrototypeModel, self).__init__()

        self.logit_fc = torch.nn.Sequential(
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, X):
        return self.logit_fc(X)


class DiscourseModel(torch.nn.Module):

    def __init__(self, joint_in_dim, joint_hid_dim, joint_out_dim):
        super(DiscourseModel, self).__init__()
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", 'cuda:0')

        self.projection_model = ProjectionModel(joint_in_dim, joint_hid_dim)
        self.prototype_model = PrototypeModel(joint_hid_dim // 2, joint_out_dim)
        # self.softmax = nn.Softmax()

    def get_image_repr(self, X):
        
        self.clip_model.to(X.device)
        self.clip_model = self.clip_model.to('cuda:0')
        return self.clip_model.encode_image(X)

    def get_text_repr(self, X):
    
        return self.clip_model.encode_text(X)

    def forward(self, high_res_images, low_res_images, texts, is_supervised):
        teach_logits = None
        stud_logits = None

        if is_supervised:
            teach_logits = self._sup_fw(high_res_images, texts)
        else:
            teach_logits, stud_logits = self._unsup_fw(high_res_images, low_res_images, texts)
        return teach_logits, stud_logits

    def _sup_fw(self, images, texts):

        image_features = self.get_image_repr(images)
        text_features = self.get_text_repr(texts)
        joint_features = torch.cat([image_features, text_features], axis=1)
        joint_features = joint_features.float()
        logits = self.projection_model(joint_features)
        orig_logits = self.prototype_model(logits).detach()

        return orig_logits

    def _unsup_fw(self, high_res_images, low_res_images, texts):

        orig_image_features = self.get_image_repr(high_res_images).detach()
        orig_text_features = self.get_text_repr(texts).detach()
        orig_joint_features = torch.cat([orig_image_features, orig_text_features], axis=1).detach()
        orig_joint_features = orig_joint_features.float()
        orig_projection_logits = self.projection_model(orig_joint_features).detach()
        orig_logits = self.prototype_model(orig_projection_logits).detach()
        orig_logits = orig_logits.detach()

        aug_images = low_res_images
        aug_image_features = self.get_image_repr(aug_images)
        aug_text_features = self.get_text_repr(texts)
        aug_joint_features = torch.cat([aug_image_features, aug_text_features], axis=1)
        aug_joint_features = aug_joint_features.float()
        aug_projection_logits = self.projection_model(aug_joint_features)
        aug_logits = self.prototype_model(aug_projection_logits)
        aug_projection_logits = aug_projection_logits.detach()
        return orig_logits, aug_logits

    @torch.no_grad()
    def distributed_sinkhorn(self, out):
        Q = torch.exp(out / 0.05).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # * args.world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        # dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(3):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            # dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

    def compute_loss(self, true_targets, pred_logits, loss, is_supervised):
        temp = 0.1
        if is_supervised:
            total_loss = loss(true_targets, pred_logits)
        else:
            true_targets = self.distributed_sinkhorn(true_targets).detach()
            # pred_logits = self.softmax(pred_logits)
            # total_loss = loss(pred_logits, true_targets)
            pred_logits /= temp
            total_loss = - torch.mean(
                torch.sum(true_targets * torch.nn.functional.log_softmax(pred_logits, dim=1), dim=1))
        return total_loss
