
# vqa
from .vqa_medvqa_2019_datamodule import VQAMEDVQA2019DataModule
from .vqa_slack_datamodule import VQASLACKDataModule
# irtr
from .irtr_roco_datamodule import IRTRROCODataModule
from .irtr_mimic_datamodule import IRTRMIMICDataModule
_datamodules = {

    # vqa
    "vqa_medvqa_2019": VQAMEDVQA2019DataModule,
    "vqa_slack" : VQASLACKDataModule,
    # irtr
    "irtr_roco": IRTRROCODataModule,

}
