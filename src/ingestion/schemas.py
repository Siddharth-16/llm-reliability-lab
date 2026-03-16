from pydantic import BaseModel
from typing import List, Optional, Dict


class Document(BaseModel):
    doc_id: str
    title: str
    text: str
    source: Optional[str] = None
    metadata: Optional[Dict] = None


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict


class QuestionSample(BaseModel):
    sample_id: str
    question: str
    ground_truth_answer: Optional[str] = None
    reference_doc_ids: Optional[List[str]] = None
    question_type: Optional[str] = None
    attack_type: Optional[str] = None