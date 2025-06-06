import torch
import random

from transformers.optimization import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from .dist_utils import all_gather
from .objectives import compute_irtr_recall_knowledge, compute_irtr_recall_no_knowledge
from ..gadgets.my_metrics import Accuracy, VQAScore, Scalar, VQARADScore, MultiClassAccuracy


def set_metrics(pl_module):
    for split in ["train", "val"]:
        for k, v in pl_module.hparams.config["loss_names"].items():
            if v <= 0:
                continue
            if k == "vqa":
                if k in ["vqa"]:
                    if split == "train":
                        setattr(pl_module, f"train_{k}_score", VQARADScore())
                        setattr(pl_module, f"train_{k}_loss", Scalar())
                    else:
                        setattr(pl_module, f"val_{k}_score", VQARADScore())
                        setattr(pl_module, f"val_{k}_loss", Scalar())
                        setattr(pl_module, f"test_{k}_score", VQARADScore())
                        setattr(pl_module, f"test_{k}_loss", Scalar())
            elif k == "nlvr2":
                if split == "train":
                    setattr(pl_module, f"train_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"train_{k}_loss", Scalar())
                else:
                    setattr(pl_module, f"dev_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"dev_{k}_loss", Scalar())
                    setattr(pl_module, f"test_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"test_{k}_loss", Scalar())
            elif k == "snli":
                if split == "train":
                    setattr(pl_module, f"train_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"train_{k}_loss", Scalar())
                else:
                    setattr(pl_module, f"dev_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"dev_{k}_loss", Scalar())
                    setattr(pl_module, f"test_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"test_{k}_loss", Scalar())
            elif k == "irtr":
                if split == "train":
                    setattr(pl_module, f"train_irtr_loss", Scalar())
                else:
                    setattr(pl_module, f"val_irtr_loss", Scalar())
                    setattr(pl_module, f"test_irtr_loss", Scalar())
            elif k == "mppd" or k == "mpfr":
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "itm":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            else:
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())


def epoch_wrapup(pl_module, test=False):
    if test:
        phase = "test"
    else:
        phase = "train" if pl_module.training else "val"
    # phase = "train" if pl_module.training else "val"
    the_metric = 0

    if (pl_module.hparams.config["get_recall_metric"] and not pl_module.training) or (phase=="test" and pl_module.hparams.config["loss_names"]["irtr"] > 0):
        if pl_module.hparams.config["kpl_meter"]:
            (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10) = compute_irtr_recall_knowledge(pl_module)
        else:
            (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10) = compute_irtr_recall_no_knowledge(pl_module)
        print((ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10), pl_module.global_step)
        pl_module.log(
            "recalls/ir_r1", ir_r1, pl_module.global_step
        )
        pl_module.log(
            "recalls/ir_r5", ir_r5, pl_module.global_step
        )
        pl_module.log(
            "recalls/ir_r10", ir_r10, pl_module.global_step
        )
        pl_module.log(
            "recalls/tr_r1", tr_r1, pl_module.global_step
        )
        pl_module.log(
            "recalls/tr_r5", tr_r5, pl_module.global_step
        )
        pl_module.log(
            "recalls/tr_r10", tr_r10, pl_module.global_step
        )
        the_metric += ir_r1.item() + tr_r1.item()

    for loss_name, v in pl_module.hparams.config["loss_names"].items():
        if v <= 0:
            continue

        value = 0

        if loss_name == "vqa":
            value = getattr(pl_module, f"{phase}_{loss_name}_score").compute()
            pl_module.log(f"{loss_name}/{phase}/score_epoch", value)
            pl_module.log(f"{loss_name}/{phase}/score_best_epoch",
                          getattr(pl_module, f"{phase}_{loss_name}_score").get_best_score())
            pl_module.log(f"{loss_name}/{phase}/close_score_best_epoch",
                          getattr(pl_module, f"{phase}_{loss_name}_score").get_best_close_score())
            pl_module.log(f"{loss_name}/{phase}/open_score_best_epoch",
                          getattr(pl_module, f"{phase}_{loss_name}_score").get_best_open_score())
            getattr(pl_module, f"{phase}_{loss_name}_score").reset()

            pl_module.log(f"{loss_name}/{phase}/loss_epoch", getattr(pl_module, f"{phase}_{loss_name}_loss").compute())
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()


            # value = getattr(pl_module, f"{phase}_{loss_name}_score").compute()
            # pl_module.log(f"{loss_name}/{phase}/score_epoch", value)
            # getattr(pl_module, f"{phase}_{loss_name}_score").reset()
            # pl_module.log(
            #     f"{loss_name}/{phase}/loss_epoch",
            #     getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            # )
            # getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "nlvr2" or loss_name == 'snli':
            if phase == "train":
                value = getattr(pl_module, f"train_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/train/accuracy_epoch", value)
                getattr(pl_module, f"train_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/train/loss_epoch",
                    getattr(pl_module, f"train_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"train_{loss_name}_loss").reset()
            else:
                value = getattr(pl_module, f"test_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/test/accuracy_epoch", value)
                getattr(pl_module, f"test_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/test/loss_epoch",
                    getattr(pl_module, f"test_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"test_{loss_name}_loss").reset()

                value = getattr(pl_module, f"dev_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/dev/accuracy_epoch", value)
                getattr(pl_module, f"dev_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/dev/loss_epoch",
                    getattr(pl_module, f"dev_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"dev_{loss_name}_loss").reset()
        elif loss_name == "irtr":
            pl_module.log(
                f"{loss_name}/{phase}/irtr_loss_epoch",
                getattr(pl_module, f"{phase}_irtr_loss").compute(),
            )
            getattr(pl_module, f"{phase}_irtr_loss").reset()
        elif loss_name == "mppd" or loss_name == "mpfr":
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "itm":
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        else:
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        the_metric += value

    pl_module.log(f"{phase}/the_metric", the_metric)


def check_non_acc_grad(pl_module):
    if pl_module.token_type_embeddings.weight.grad is None:
        return True
    else:
        grad = pl_module.token_type_embeddings.weight.grad
        return (grad.sum() == 0).item()


def set_task(pl_module):
    pl_module.current_tasks = [
        k for k, v in pl_module.hparams.config["loss_names"].items() if v > 0
    ]
    return

def set_schedule(pl_module):
    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config["weight_decay"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    head_names = ["vqa_classifier", "itm_score","rank_output"]
    cross_modal_names = ['cross_modal']
    lr_mult_head = pl_module.hparams.config["lr_mult_head"]
    lr_mult_cross_modal = pl_module.hparams.config["lr_mult_cross_modal"]
    end_lr = pl_module.hparams.config["end_lr"]
    decay_power = pl_module.hparams.config["decay_power"]
    optim_type = pl_module.hparams.config["optim_type"]
    if pl_module.hparams.config["kpl_meter"]:
        adapter_related_names = ["adapter", "SAdapter"]
        backbone_lr = pl_module.hparams.config["backbone_lr"]
        optimizer_grouped_parameters = [
            # only params
            {   #backbone-weight
                "params": [
                    p
                    for n, p in pl_module.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                    and p.requires_grad
                ],
                "weight_decay": wd,
                "lr": backbone_lr,
            },
            {   #backbone-bias
                "params": [
                    p
                    for n, p in pl_module.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                    and p.requires_grad
                ],
                "weight_decay": 0.0,
                "lr": backbone_lr,
            },
            {   #head-weight
                "params": [
                    p
                    for n, p in pl_module.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and any(bb in n for bb in head_names)
                    and not any(ht in n for ht in adapter_related_names)
                ],
                "weight_decay": wd,
                "lr": lr * lr_mult_head,
            },
            {   #head-bias
                "params": [
                    p
                    for n, p in pl_module.named_parameters()
                    if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
                    and not any(ht in n for ht in adapter_related_names)
                ],
                "weight_decay": 0.0,
                "lr": lr * lr_mult_head,
            },
        ]
    else:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in pl_module.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                    and not any(ht in n for ht in cross_modal_names)
                ],
                "weight_decay": wd,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in pl_module.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                    and not any(ht in n for ht in cross_modal_names)
                ],
                "weight_decay": 0.0,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in pl_module.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and any(bb in n for bb in head_names)
                    and not any(ht in n for ht in cross_modal_names)
                ],
                "weight_decay": wd,
                "lr": lr * lr_mult_head,
            },
            {
                "params": [
                    p
                    for n, p in pl_module.named_parameters()
                    if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
                    and not any(ht in n for ht in cross_modal_names)
                ],
                "weight_decay": 0.0,
                "lr": lr * lr_mult_head,
            },
            {
                "params": [
                    p
                    for n, p in pl_module.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                    and any(ht in n for ht in cross_modal_names)
                ],
                "weight_decay": wd,
                "lr": lr * lr_mult_cross_modal,
            },
            {
                "params": [
                    p
                    for n, p in pl_module.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                    and any(ht in n for ht in cross_modal_names)
                ],
                "weight_decay": 0.0,
                "lr": lr * lr_mult_cross_modal,
            },
        ]
    
    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    if pl_module.trainer.max_steps is None:
        max_steps = (
            len(pl_module.trainer.datamodule.train_dataloader())
            * pl_module.trainer.max_epochs
            // pl_module.trainer.accumulate_grad_batches
        )
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(pl_module.hparams.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )