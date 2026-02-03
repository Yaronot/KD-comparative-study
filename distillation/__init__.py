"""Knowledge distillation methods."""

from .vanilla_kd import distillation_loss
from .fitnet import extract_teacher_hint, train_fitnet_stage1
from .rkd import compute_distance_matrix
