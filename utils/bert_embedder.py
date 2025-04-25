import torch
from transformers import AutoTokenizer, AutoModel
from constants import BERT_MODEL_NAME, MAX_TOKEN_LENGTH, DEVICE

# Load tokenizer and model once (shared across uses)
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)
bert_model.eval()  # no gradients needed for inference

@torch.no_grad()
def encode_texts_with_bert(text_list):
    """
    Takes a list of strings (comments/submissions) and returns [CLS] token embeddings from BERT.
    """
    encoded_input = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=MAX_TOKEN_LENGTH,
        return_tensors="pt"
    ).to(DEVICE)

    outputs = bert_model(**encoded_input)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    return cls_embeddings.cpu()  # return on CPU for downstream use
