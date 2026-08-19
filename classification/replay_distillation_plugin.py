# replay_distillation_plugin.py

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from avalanche.training.plugins import SupervisedPlugin


class ReplayDistillationPlugin(SupervisedPlugin):

    def __init__(
            self,
            lambda_kd=0.2,
            lambda_relation=0.2,
            lambda_proto=0.1,
            lambda_semantic=0.0,
            temperature=2.0,
            parent_global=None,
            batch_size_proto=64,
    ):
        super().__init__()

        self.lambda_kd = lambda_kd
        self.lambda_relation = lambda_relation
        self.lambda_proto = lambda_proto
        self.lambda_semantic = lambda_semantic

        self.temperature = temperature
        self.batch_size_proto = batch_size_proto

        if parent_global is None:
            parent_global = {}

        self.parent_global = parent_global

        # teacher
        self.old_model = None

        # {class_id: prototype}
        self.class_prototypes = {}

    ####################################################################
    # Feature KD
    ####################################################################
    def kd_loss(
            self,
            fs,
            ft
    ):

        fs = F.normalize(fs, dim=1)
        ft = F.normalize(ft, dim=1)

        return (
                1
                -
                F.cosine_similarity(
                    fs,
                    ft.detach(),
                    dim=1
                ).mean()
        )

    ####################################################################
    # Relation Preserve Loss
    ####################################################################
    def relation_loss(
            self,
            fs,
            ft
    ):

        fs = F.normalize(fs, dim=1)
        ft = F.normalize(ft, dim=1)

        Rs = torch.matmul(
            fs,
            fs.T
        )

        Rt = torch.matmul(
            ft,
            ft.T
        )

        return F.mse_loss(
            Rs,
            Rt.detach()
        )

    ####################################################################
    # Prototype Loss
    ####################################################################
    def prototype_loss(
            self,
            feats,
            labels
    ):
        # if strategy.clock.train_exp_counter < 1:
        #     return torch.tensor(
        #         0.,
        #         device=feats.device
        #     )

        if len(self.class_prototypes) == 0:
            return torch.tensor(
                0.0,
                device=feats.device
            )

        loss = torch.tensor(
            0.0,
            device=feats.device
        )

        n_cls = 0

        for c in torch.unique(labels):

            c_int = int(c)

            if c_int not in self.class_prototypes:
                continue

            mask = labels == c

            cls_feat = feats[mask]

            if len(cls_feat) == 0:
                continue

            proto_new = cls_feat.mean(0)

            proto_new = F.normalize(
                proto_new.unsqueeze(0),
                dim=1
            ).squeeze(0)

            proto_old = self.class_prototypes[
                c_int
            ].to(feats.device)

            loss += (
                    1
                    -
                    F.cosine_similarity(
                        proto_new.unsqueeze(0),
                        proto_old.unsqueeze(0),
                        dim=1
                    ).mean()
            )

            n_cls += 1

        if n_cls > 0:
            loss = loss / n_cls

        return loss

    ####################################################################
    # Semantic KD
    ####################################################################
    def semantic_loss(
            self,
            logits
    ):

        if self.lambda_semantic == 0:
            return torch.tensor(
                0.0,
                device=logits.device
            )

        if len(self.parent_global) == 0:
            return torch.tensor(
                0.0,
                device=logits.device
            )

        B, C = logits.shape

        active_classes = list(
            range(C)
        )

        global_to_local = {
            g: i
            for i, g in enumerate(active_classes)
        }

        target_logits = logits.detach().clone()

        valid_pairs = 0

        for child, parent in self.parent_global.items():

            if child not in global_to_local:
                continue

            if parent not in global_to_local:
                continue

            child_l = global_to_local[child]
            parent_l = global_to_local[parent]

            target_logits[:, parent_l] += (
                    0.1
                    *
                    target_logits[:, child_l]
            )

            valid_pairs += 1

        if valid_pairs == 0:
            return torch.tensor(
                0.0,
                device=logits.device
            )

        student = F.log_softmax(
            logits / self.temperature,
            dim=1
        )

        teacher = F.softmax(
            target_logits / self.temperature,
            dim=1
        )

        loss = F.kl_div(
            student,
            teacher.detach(),
            reduction="batchmean"
        )

        loss *= self.temperature ** 2

        return loss

    # def before_training_exp(self, strategy, **kwargs):
    #
    #     exp_id = strategy.clock.train_exp_counter
    #
    #     # Task0 train full
    #     if exp_id < 1:
    #         return
    #
    #     # chỉ freeze 1 lần
    #     if getattr(self, "_frozen", False):
    #         return
    #
    #
    #     backbone = strategy.model.feature_extractor
    #
    #     print(f"Exp {exp_id}: freeze early layers")
    #
    #     for p in backbone.conv1.parameters():
    #         p.requires_grad = False
    #
    #     for p in backbone.bn1.parameters():
    #         p.requires_grad = False
    #
    #     for p in backbone.layer1.parameters():
    #         p.requires_grad = False
    #
    #     # rebuild optimizer
    #     old_opt = strategy.optimizer
    #
    #     lr = old_opt.param_groups[0]["lr"]
    #     momentum = old_opt.param_groups[0].get("momentum", 0.9)
    #     weight_decay = old_opt.param_groups[0].get("weight_decay", 5e-4)
    #
    #     strategy.optimizer = torch.optim.SGD(
    #         filter(
    #             lambda p: p.requires_grad,
    #             strategy.model.parameters()
    #         ),
    #         lr=lr,
    #         momentum=momentum,
    #         weight_decay=weight_decay
    #     )
    #
    #     self._frozen = True
    #
    #     # debug
    #     n_train = sum(
    #         p.numel()
    #         for p in strategy.model.parameters()
    #         if p.requires_grad
    #     )
    #
    #     n_total = sum(
    #         p.numel()
    #         for p in strategy.model.parameters()
    #     )
    #
    #     print(
    #         f"Trainable params: "
    #         f"{n_train:,}/{n_total:,}"
    #     )


    ####################################################################
    # SAVE TEACHER + PROTOTYPES
    ####################################################################
    def after_training_exp(
            self,
            strategy,
            **kwargs
    ):
        exp_id = strategy.experience.current_experience

        if exp_id == 0:
            # chỉ lưu teacher
            self.old_model = deepcopy(strategy.model)
            self.old_model.eval()

            for p in self.old_model.parameters():
                p.requires_grad = False

            return

        ########################################
        # save teacher
        ########################################

        self.old_model = deepcopy(
            strategy.model
        )

        self.old_model.eval()

        for p in self.old_model.parameters():
            p.requires_grad = False

        ########################################
        # save prototypes
        ########################################

        dataset = strategy.experience.dataset

        loader = DataLoader(
            dataset.eval(),
            batch_size=self.batch_size_proto,
            shuffle=False
        )

        proto_dict = {}

        with torch.no_grad():

            for x, y, _ in loader:

                x = x.to(strategy.device)

                feat = (
                    strategy.model
                    .feature_extractor(x)
                )

                feat = F.normalize(
                    feat,
                    dim=1
                )

                y = y.cpu()

                for c in torch.unique(y):

                    c_int = int(c)

                    mask = (
                            y == c
                    )

                    mask = mask.to(
                        feat.device
                    )

                    cls_feat = feat[mask]

                    if len(cls_feat) == 0:
                        continue

                    proto = cls_feat.mean(0)

                    proto = F.normalize(
                        proto.unsqueeze(0),
                        dim=1
                    ).squeeze(0)

                    proto_dict[c_int] = (
                        proto.detach().cpu()
                    )

        self.class_prototypes = proto_dict

    ####################################################################
    # ADD EXTRA LOSS
    ####################################################################
    def before_backward(
            self,
            strategy,
            **kwargs
    ):
        # Experience 0:
        # giữ nguyên ICaRL gốc
        # if strategy.clock.train_exp_counter == 0:
        #     return
        # Task1 = baseline
        if strategy.clock.train_exp_counter < 1:
            return

        # print(
        #     "Exp:",
        #     strategy.clock.train_exp_counter,
        #     "Extra loss enabled"
        # )
        if self.old_model is None:
            return

        x = strategy.mb_x
        y = strategy.mb_y
        logits = strategy.mb_output

        ########################################
        # student features
        ########################################

        fs = (strategy.model.feature_extractor(x) )

        fs = F.normalize( fs, dim=1 )

        ########################################
        # teacher features
        ########################################

        with torch.no_grad():

            ft = (
                self.old_model
                .feature_extractor(x)
            )

            ft = F.normalize(
                ft,
                dim=1
            )

        ########################################
        # losses
        ########################################

        loss_kd = self.kd_loss(
            fs,
            ft
        )

        loss_relation = self.relation_loss(
            fs,
            ft
        )

        loss_proto = self.prototype_loss(
            fs,
            y
        )

        loss_semantic = self.semantic_loss(
            logits
        )

        ########################################
        # total
        ########################################

        extra_loss = (
                self.lambda_kd
                * loss_kd
                +
                self.lambda_relation
                * loss_relation
                +
                self.lambda_proto
                * loss_proto
                +
                self.lambda_semantic
                * loss_semantic
        )
        # print('extra_loss: ', extra_loss)
        # print('strategy.loss: ', strategy.loss)

        strategy.loss += extra_loss


        ########################################
        # logging
        ########################################

        strategy.loss_kd = float(
            loss_kd.detach()
        )

        strategy.loss_relation = float(
            loss_relation.detach()
        )

        strategy.loss_proto = float(
            loss_proto.detach()
        )

        strategy.loss_semantic = float(
            loss_semantic.detach()
        )