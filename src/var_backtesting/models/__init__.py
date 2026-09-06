"""Five original dissertation models."""

from .garch import garch_normal_var, garch_t_var
from .historical import historical_var
from .parametric import normal_var, student_t_var

MODELS = {
    "historical": historical_var,
    "normal": normal_var,
    "student_t": student_t_var,
    "garch_normal": garch_normal_var,
    "garch_t": garch_t_var,
}
