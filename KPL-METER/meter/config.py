from sacred import Experiment

ex = Experiment("METER")


def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "vqa": 0,
        "vcr": 0,
        "vcr_qar": 0,
        "nlvr2": 0,
        "irtr": 0,
        "contras": 0,
        "snli": 0,
        "cls": 0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "meter"
    seed = 42
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_size = 224
    patch_size = 32
    draw_false_image = 1
    image_only = False
    resolution_before = 224

    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 40
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    num_top_layer = 6
    input_image_embed_size = 768
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    hidden_size = 768
    num_heads = 12
    num_layers = 6
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-5
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 100000
    warmup_steps = 10000
    end_lr = 0
    lr_mult_head = 5  # multiply lr for downstream heads
    lr_mult_cross_modal = 5  # multiply lr for the cross-modal module

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 8
    num_nodes = 1
    load_path = ""
    num_workers = 16
    precision = 16

    max_num_ents = 24
    label_column_name = ""
    num_batch_multi = 2
    model_arch = ""

    # adapter related
    monitor = "" # for checkpoint_callback
    # fine-tune strategies
    linear = False
    top_k = False
    # trained model save to 
    saved_path = "./finetune/meter"

    # adapter_related
    backbone_lr = 1e-3
    adapter_factor = 4
    multi_adapter_dim = 128
    clip_add_adapter = False
    roberta_add_adapter = False
    multi_adapter_factor = 4

    # == irtr
    irtr_recall_batch_size = 128
    
    return_weights = False
    # knowledge
    kpl_meter = False
    prompt_text = False

@ex.named_config
def task_finetune_vqa_medvqa_2019():
    exp_name = "kpl_meter_task_finetune_vqa_medvqa_2019"
    datasets = ["vqa_medvqa_2019"]
    loss_names = _loss_names({"vqa": 1})
    # monitor = "vqa/val/score"
    batch_size = 32
    warmup_steps = 0.1
    learning_rate = 5e-5
    train_transform_keys = ["m3ae_clip_transform"]
    val_transform_keys = ["m3ae_clip_transform"]
    image_size = 384
    vqav2_label_size = 79
    monitor = "vqa/val/score_best_epoch"
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    lr_mult_head = 50
    lr_mult_cross_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 32
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    input_image_embed_size = 768


@ex.named_config
def task_finetune_vqa_slack():
    exp_name = "kpl_meter_task_finetune_vqa_slack"
    datasets = ["vqa_slack"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 32
    warmup_steps = 0.1
    learning_rate = 5e-5
    train_transform_keys = ["m3ae_clip_transform"]
    val_transform_keys = ["m3ae_clip_transform"]
    image_size = 384
    vqav2_label_size = 222
    monitor = "vqa/val/score_best_epoch"

    max_epoch = 15
    max_steps = None
    lr_mult_head = 50
    lr_mult_cross_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 32
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    input_image_embed_size = 768

@ex.named_config
def task_finetune_irtr_roco():
    exp_name = "kpl_meter_task_finetune_irtr_roco"
    datasets = ["irtr_roco"]
    loss_names = _loss_names({"irtr": 1})
    monitor = "irtr/val/irtr_loss_epoch"
    batch_size = 32
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = False
    draw_false_text = 15
    learning_rate = 5e-6
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    tokenizer = "bert-base-uncased"
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["m3ae_clip_transform"]
    val_transform_keys = ["m3ae_clip_transform"]
    input_image_embed_size = 768
    image_size = 224

@ex.named_config
def task_mlm_itm_clip_bert():
    exp_name = "mlm_itm"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096
    max_epoch = 10
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True

    vocab_size = 30522
    max_text_len = 50
    image_size = 224
    tokenizer = "bert-base-uncased"
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    learning_rate = 1e-5
    val_check_interval = 1.0
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    num_top_layer = 6
    hidden_size = 768
    num_heads = 12

@ex.named_config
def task_finetune_nlvr2_clip_bert():
    exp_name = "finetune_nlvr2"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-5
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 50
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 288

@ex.named_config
def task_finetune_vqa_clip_bert():
    exp_name = "finetune_vqa"
    datasets = ["vqa"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-6
    val_check_interval = 0.1
    lr_mult_head = 50
    lr_mult_cross_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 50
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 576

@ex.named_config
def task_finetune_irtr_coco_clip_bert():
    exp_name = "finetune_irtr_coco"
    datasets = ["coco"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 5e-6
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    tokenizer = "bert-base-uncased"
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 384

@ex.named_config
def task_finetune_irtr_f30k_clip_bert():
    exp_name = "finetune_irtr_f30k"
    datasets = ["f30k"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 5e-6
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    tokenizer = "bert-base-uncased"
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 384

@ex.named_config
def task_finetune_snli_clip_bert():
    exp_name = "finetune_snli"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    batch_size = 64
    max_epoch = 5
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 50
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 384


# Named configs for "etc" which are orthogonal to "env" and "task", need to be added at the end

# vision encoder
@ex.named_config
def swin32_base224():
    vit = "swin_base_patch4_window7_224_in22k"
    patch_size = 32
    image_size = 224
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    input_image_embed_size = 1024
    resolution_before = 224

@ex.named_config
def swin32_base384():
    vit = "swin_base_patch4_window12_384_in22k"
    patch_size = 32
    image_size = 384
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    input_image_embed_size = 1024
    resolution_before = 384

@ex.named_config
def swin32_large384():
    vit = "swin_large_patch4_window12_384_in22k"
    patch_size = 32
    image_size = 384
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    input_image_embed_size = 1536
    resolution_before = 384

@ex.named_config
def clip32():
    vit = 'ViT-B/32'
    image_size = 224
    patch_size = 32
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768

@ex.named_config
def clip16():
    vit = 'ViT-B/16'
    image_size = 224
    patch_size = 16
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768

@ex.named_config
def clip16_adapter():
    vit = 'ViT-B/16'
    image_size = 224
    patch_size = 16
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    clip_add_adapter = True
    roberta_add_adapter = True
@ex.named_config
def clip16_kpl():
    vit = 'ViT-B/16'
    image_size = 224
    patch_size = 16
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    kpl_meter = True




@ex.named_config
def clip16_vpt():
    vit = 'ViT-B/16'
    image_size = 224
    patch_size = 16
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    visual_vpt = True

# text encoder
@ex.named_config
def text_roberta():
    tokenizer = "roberta-base"
    vocab_size = 50265
    input_text_embed_size = 768

@ex.named_config
def text_roberta_large():
    tokenizer = "roberta-large"
    vocab_size = 50265
    input_text_embed_size = 1024

# random augmentation
@ex.named_config
def imagenet_randaug():
    train_transform_keys = ["imagenet_randaug"]

@ex.named_config
def clip_randaug():
    train_transform_keys = ["clip_randaug"]
