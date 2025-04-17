from .vg_caption_datamodule import VisualGenomeCaptionDataModule
from .f30k_caption_karpathy_datamodule import F30KCaptionKarpathyDataModule
from .coco_caption_karpathy_datamodule import CocoCaptionKarpathyDataModule
from .conceptual_caption_datamodule import ConceptualCaptionDataModule
from .sbu_datamodule import SBUCaptionDataModule
from .wikibk_datamodule import WikibkDataModule
from .vqav2_datamodule import VQAv2DataModule
from .nlvr2_datamodule import NLVR2DataModule
from .vqa_slack_datamodule import VQASLACKDataModule
from .vqa_medvqa_2019_datamodule import VQAMEDVQA2019DataModule
from .irtr_roco_datamodule import IRTRROCODataModule
_datamodules = {
    "vg": VisualGenomeCaptionDataModule,
    "f30k": F30KCaptionKarpathyDataModule,
    "coco": CocoCaptionKarpathyDataModule,
    "gcc": ConceptualCaptionDataModule,
    "sbu": SBUCaptionDataModule,
    "wikibk": WikibkDataModule,
    "vqa": VQAv2DataModule,
    "nlvr2": NLVR2DataModule,
    "vqa_slack": VQASLACKDataModule,
    "vqa_medvqa_2019": VQAMEDVQA2019DataModule,
    "irtr_roco": IRTRROCODataModule,

}
