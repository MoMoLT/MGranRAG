from pydantic import BaseModel
from typing import List


# 定义单个事实条目模型
class FactItem(BaseModel):
    sentence_id: str
    sentence: str
    clue: str
    keywords: List[str]
    # relevance_level: str
    # clue: str


# 定义顶层列表模型
class FactList(BaseModel):
    reasoning: str
    findings: List[FactItem]

# class FactItem(BaseModel):
#     id: str
#     content: str
#     resolved_sentence: str
#     keywords: List[str]
#     # relevance_level: str
#     # clue: str
#
#
# # 定义顶层列表模型
# class FactList(BaseModel):
#     facts: List[FactItem]



class ClueItem(BaseModel):
    id: str
    content: str
    keywords: List[str]
    clue: str


class ClueList(BaseModel):
    clues: List[ClueItem]


class InferClueItem(BaseModel):
    id: str
    content: str
    keywords: List[str]


class InferList(BaseModel):
    facts: List[InferClueItem]

# class InferRelevanceItem(BaseModel):
#     category: str
#     reason: str
#
#
# class InferClueItem(BaseModel):
#     source_id: int
#     related_entities: List[str]
#     fact_relevance: InferRelevanceItem
#
#
# class InferList(BaseModel):
#     reasoning_steps: str
#     intermediate_facts: List[InferClueItem]
#     derived_answer: str
