import re
from typing import List, Callable
from transformers import pipeline


# 初始化NER模型（中文）
ner_pipeline = pipeline(
    "ner",
    model="ckiplab/bert-base-chinese-ner",
    grouped_entities=True
)


def ner_mask(text: str) -> str:
    entities = ner_pipeline(text)
    spans = []
    for ent in entities:
        label = ent["entity_group"]
        start = ent["start"]
        end = ent["end"]

        if label == "PER":
            spans.append((start, end, "[NAME]"))
        elif label == "LOC":
            spans.append((start, end, "[PLACE]"))

    # 按长度优先
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # 去重叠
    filtered_spans = []
    last_end = -1
    for start, end, tag in spans:
        if start >= last_end:
            filtered_spans.append((start, end, tag))
            last_end = end

    result = []
    last_idx = 0
    for start, end, tag in filtered_spans:
        result.append(text[last_idx:start])
        result.append(tag)
        last_idx = end
    result.append(text[last_idx:])

    return "".join(result)

class DesensitizationPipeline:
    def __init__(self):
        self.steps: List[Callable[[str], str]] = []

    def add_step(self, func: Callable[[str], str]):
        self.steps.append(func)

    def run(self, text: str) -> str:
        for step in self.steps:
            text = step(text)
        return text

def normalize_text(text: str) -> str:
    # 去除多余空格、统一符号（简单示例）
    return text.strip()

# 高确定性规则
def mask_phone(text: str) -> str:
    return re.sub(r'1[3-9]\d{9}', '[PHONE]', text)


def mask_email(text: str) -> str:
    return re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '[EMAIL]', text)


# 中确定性
def mask_address(text: str) -> str:
    return re.sub(
        r'(居住于|现居住于|现居于|地址)([\u4e00-\u9fa5A-Za-z0-9]+)',
        r'\1[PLACE]',
        text
    )



# 低确定性
def mask_name(text: str) -> str:
    # 只匹配句首 or 标点后
    return re.sub(
        r'(?:(?<=^)|(?<=[，。！？]))([\u4e00-\u9fa5]{2,3})(的)',
        r'[NAME]\2',
        text
    )

# 后处理
def clean_punctuation(text: str) -> str:
    # 可选：规范标点
    return text

# 完整处理
def build_pipeline():
    pipeline = DesensitizationPipeline()

    pipeline.add_step(normalize_text)

    # 高确定性优先处理（不会误伤）
    pipeline.add_step(mask_phone)
    pipeline.add_step(mask_email)

    # NER（主力识别）
    pipeline.add_step(ner_mask)

    # regex兜底（简单规则的模式匹配）
    pipeline.add_step(mask_address)
    pipeline.add_step(mask_name)

    pipeline.add_step(clean_punctuation)

    return pipeline

# 测试
if __name__ == "__main__":
    text = "小明的邮箱是test@gmail.com，电话是13312311111，现在居住于重庆两江新区的xxx小区。"
    pipeline = build_pipeline()

    print("处理前:", text)
    print("处理后:", pipeline.run(text))
