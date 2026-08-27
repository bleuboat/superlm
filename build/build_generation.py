import re
from pathlib import Path

from . import templater

generation = Path("src") / "superlm" / "generation"
utils_file = generation / "utils.py"
logits_process_file = generation / "logits_processor.py"
stopping_criteria_file = generation / "stopping_criteria.py"

logits_process = stopping_criteria = ""

class_pattern = re.compile(r"^class ([A-Za-z]*)", re.MULTILINE)

for cls in class_pattern.findall(logits_process_file.read_text("utf-8")):
    if cls != "LogitsProcessor":
        logits_process += f"{cls},\n"

for cls in class_pattern.findall(stopping_criteria_file.read_text("utf-8")):
    if cls != "StoppingCriteria":
        stopping_criteria += f"{cls},\n"

templater.apply(utils_file)
