import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import random
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings, BertModel, BertEncoder, BertLayer
from .multi_modal_modules.bert_model import BertCrossLayer
from . import heads, objectives, meter_utils
from transformers import RobertaConfig, RobertaModel
from .clip_modules.clip_model import build_model, adapt_position_encoding
from .my_roberta.modeling_roberta_adapter import RobertaModel_add_adapter
from .multi_modal_modules.bert_model_prompt import BertCrossLayer_with_knowledge
from .adapter_model import SAdapter_module
class METERTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.is_clip= (not 'swin' in config['vit'])

        self.log_num = 0

        if 'roberta' in config['tokenizer']:
            bert_config = RobertaConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
        else:
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )

        resolution_after=config['image_size']

        self.cross_modal_text_transform = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.cross_modal_text_transform.apply(objectives.init_weights)
        self.cross_modal_image_transform = nn.Linear(config['input_image_embed_size'], config['hidden_size'])
        self.cross_modal_image_transform.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                if self.is_clip:
                    build_model(config['vit'], resolution_after=resolution_after,)
                else:
                    pass # swin
                    # getattr(swin, self.hparams.config["vit"])(
                    #     pretrained=True, config=self.hparams.config,
                    # )

                if 'roberta' in config['tokenizer']:
                    RobertaModel.from_pretrained(config['tokenizer'])
                else:
                    raise NotImplementedError("BertModel")
                    # BertModel.from_pretrained(config['tokenizer'])

            torch.distributed.barrier()

        if self.is_clip:
            self.vit_model = build_model(config['vit'], resolution_after=resolution_after,  config=self.hparams.config, uniadapters=self.uniAdapters)
        else:
            raise NotImplementedError("swin")
            self.vit_model = getattr(swin, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config, 
            )
            self.avgpool = nn.AdaptiveAvgPool1d(1)

        # ===================== Text Encoder =====================
        if 'roberta' in config['tokenizer']:
            if config["roberta_add_adapter"]:
                self.text_transformer = RobertaModel_add_adapter.from_pretrained(config['tokenizer'], adapter_factor=config['adapter_factor'])
            else:
                self.text_transformer = RobertaModel.from_pretrained(config['tokenizer'])
        else:
            self.text_transformer = BertModel.from_pretrained(config['tokenizer'])
        
        # ======= SAdapter Config =======
        self.SAdapter = None
        if config["kpl_meter"]:
            self.SAdapter = nn.ModuleList(
                [SAdapter_module(d_model=config["hidden_size"], middle_dim=config["multi_adapter_dim"])
            for i in range(config['num_top_layer'])])
        # ======= End SAdapter Config =======


        
        
        if config["kpl_meter"]:
            self.cross_modal_text_layers =  nn.ModuleList([BertCrossLayer_with_knowledge(bert_config) for idx in range(config['num_top_layer'])])
            self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer_with_knowledge(bert_config) for idx in range(config['num_top_layer'])])
        else:
            self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
            self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])

        
        self.cross_modal_image_layers.apply(objectives.init_weights)
        self.cross_modal_text_layers.apply(objectives.init_weights)


        self.cross_modal_image_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_image_pooler.apply(objectives.init_weights)
        self.cross_modal_text_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_text_pooler.apply(objectives.init_weights)
        
        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0 or self.hparams.config["loss_names"]["irtr"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"]*2)
            self.itm_score.apply(objectives.init_weights)

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["cls"] > 0:
            if self.hparams.config["label_column_name"] != "":
                cs = self.hparams.config["melinda_label_size"][self.hparams.config["label_column_name"]]
            else:
                cs = self.hparams.config["vqav2_label_size"]
            self.cls_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cs),
            )
            self.cls_classifier.apply(objectives.init_weights)
        
        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if self.is_clip:
                state_dict = adapt_position_encoding(state_dict, after=resolution_after, patch_size=self.hparams.config['patch_size'])
            else:
                pass # swin
            self.load_state_dict(state_dict, strict=False)


        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 4, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["snli"] > 0:
            self.snli_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 3),
            )
            self.snli_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs*2, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False
        
        meter_utils.set_metrics(self)
        self.current_tasks = list()
        
        if self.hparams.config["kpl_meter"] or self.hparams.config["use_knowledge"]:
            for param in self.parameters():
                param.requires_grad = False
            need_grads = ["vqa_classifier", "adapter", "cls_classifier", "rank_output","SAdapter", ]
            for name, param in self.named_parameters():
                for key in need_grads: 
                    print(key, name)
                    if key in name:
                        param.requires_grad = True

        # ===================== load downstream (test_only) ======================
        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if self.is_clip:
                state_dict = adapt_position_encoding(state_dict, after=resolution_after, patch_size=self.hparams.config['patch_size'])
            else:
                pass # swin
                # state_dict = swin_adapt_position_encoding(state_dict, after=resolution_after, before=config['resolution_before'])
            self.load_state_dict(state_dict, strict=False)

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        img=None,
    ):
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                imgkey = f"image_{image_token_type_idx - 1}"
            else:
                imgkey = "image"
            img = batch[imgkey][0]

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]

        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        device = text_embeds.device
        input_shape = text_masks.size()
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, device)
        for layer in self.text_transformer.encoder.layer:
            text_embeds = layer(text_embeds, extend_text_masks)[0]
        text_embeds = self.cross_modal_text_transform(text_embeds)

        image_embeds = self.vit_model(img)
        image_embeds = self.cross_modal_image_transform(image_embeds)
        image_masks = torch.ones((image_embeds.size(0), image_embeds.size(1)), dtype=torch.long, device=device)
        extend_image_masks = self.text_transformer.get_extended_attention_mask(image_masks, image_masks.size(), device)

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        x, y = text_embeds, image_embeds
        retunrned_attentions = None
        output_attentions = False
        if self.hparams.config["return_weights"]:
            retunrned_attentions = {"text2image": [], "image2text": [], "ent2image": []}
            output_attentions = True
        for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
            x1 = text_layer(x, y, extend_text_masks, extend_image_masks,output_attentions=output_attentions)
            y1 = image_layer(y, x, extend_image_masks, extend_text_masks,output_attentions=output_attentions)
            x, y = x1[0], y1[0]

            if self.hparams.config["return_weights"]:
                retunrned_attentions["text2image"].append(x1[1:])
                retunrned_attentions["image2text"].append(y1[1:])
        text_feats, image_feats = x, y
        cls_feats_text = self.cross_modal_text_pooler(x)
        if self.is_clip:
            cls_feats_image = self.cross_modal_image_pooler(y)
        else:
            avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
            cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)
        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "retunrned_attentions" : retunrned_attentions,
        }


        return ret
    
    def infer_prompt(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        img=None,
    ):
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                imgkey = f"image_{image_token_type_idx - 1}"
            else:
                imgkey = "image"
            img = batch[imgkey][0]

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        device = text_ids.device
        # ====== Prompt Text (Knowledge) ======
        if self.hparams.config["prompt_text"]:
            prompt_text_ids = batch["prompt_str_ids"]
            prompt_text_embeds = self.text_transformer.embeddings(input_ids=prompt_text_ids)
            prompt_text_masks = batch["prompt_str_masks"]
            device = prompt_text_embeds.device
            prompt_input_shape = prompt_text_masks.size()
            prompt_extend_text_masks = self.text_transformer.get_extended_attention_mask(prompt_text_masks, prompt_input_shape, device)
            for layer in self.text_transformer.encoder.layer:
                prompt_text_embeds = layer(prompt_text_embeds, prompt_extend_text_masks, need_adapter=False)[0]
            prompt_text_embeds = self.cross_modal_text_transform(prompt_text_embeds)
        else:
            prompt_text_embeds = None
            prompt_extend_text_masks = None
        # ====== End Prompt Text ======

        # ====== Text ======
        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        device = text_embeds.device
        input_shape = text_masks.size()
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, device)
        for layer in self.text_transformer.encoder.layer:
            text_embeds = layer(text_embeds, extend_text_masks)[0]
        text_embeds = self.cross_modal_text_transform(text_embeds)
        # ====== End Text ======

        # ====== Image ======
        image_embeds = self.vit_model(img)
        image_embeds = self.cross_modal_image_transform(image_embeds)
        image_masks = torch.ones((image_embeds.size(0), image_embeds.size(1)), dtype=torch.long, device=device)
        extend_image_masks = self.text_transformer.get_extended_attention_mask(image_masks, image_masks.size(), device)
        # ====== End Image ======

        # ====== token_type ======
        ## text
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))
        ## prompt text (Knowledge)
        if self.hparams.config["prompt_text"]:
            prompt_text_embeds = prompt_text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))
        else:
            prompt_text_embeds = None
        ## image
        image_embeds = image_embeds + self.token_type_embeddings(
            torch.full_like(image_masks, image_token_type_idx)
        )
        # ====== End token_type ======

        x, y, z = text_embeds, image_embeds, prompt_text_embeds

        idx=0
        retunrned_attentions = None
        output_attentions = False
        for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
            # prompt text (Knowledge)
            x1, z1 = text_layer(x, y, extend_text_masks, extend_image_masks,output_attentions=output_attentions, z=z, prompt_str_mask=prompt_extend_text_masks, mode="text", adapter=self.SAdapter[idx])

            y1 = image_layer(y, x, extend_image_masks, extend_text_masks,output_attentions=output_attentions, z=z, prompt_str_mask=prompt_extend_text_masks, mode="image", adapter=self.SAdapter[idx])
            x, y, z = x1[0], y1[0], z1[0]
            idx+=1
            if self.hparams.config["return_weights"]:
                retunrned_attentions["text2image"].append(x1[1:])
                retunrned_attentions["image2text"].append(y1[1:])
                retunrned_attentions["ent2image"].append(z1[1:])
        text_feats, image_feats = x, y

        cls_feats_text = self.cross_modal_text_pooler(x)
        if self.is_clip:
            cls_feats_image = self.cross_modal_image_pooler(y)
        else:
            avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
            cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)

        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "retunrned_attentions" : retunrned_attentions,
        }


        return ret

    def forward(self, batch, test=False):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch, test))
        
        # CLS
        if "cls" in self.current_tasks:
            ret.update(objectives.compute_cls(self, batch, test))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # SNLI Visual Entailment
        if "snli" in self.current_tasks:
            ret.update(objectives.compute_snli(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            if self.hparams.config["kpl_meter"]:
                ret.update(objectives.computer_irtr_knowledge(self, batch, test))
            else:
                ret.update(objectives.computer_irtr_no_knowledge(self, batch, test))

        return ret

    def training_step(self, batch, batch_idx):
        meter_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        meter_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        meter_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        meter_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        meter_utils.set_task(self)
        output = self(batch, test=True)
    def test_epoch_end(self, outs):
        meter_utils.epoch_wrapup(self, test=True)
    def configure_optimizers(self):
        return meter_utils.set_schedule(self)
