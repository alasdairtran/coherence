from allennlp.training.trainer_base import TrainerBase

from .trainer import TrainerFP16


@TrainerBase.register("trainer_fp16_single")
class TrainerF16SingleTask(TrainerFP16):
    pass
