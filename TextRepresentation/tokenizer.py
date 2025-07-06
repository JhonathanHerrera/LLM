#!/usr/bin/env python3

import time
from transformers import AutoTokenizer
import sentencepiece as spm
import spacy
import nltk
from nltk.tokenize import TreebankWordTokenizer
from tokenizers import Tokenizer

nltk.download('punkt')

text = "こんにちは、元気ですか？" * 1000  

# -------------------- AutoTokenizer --------------------
start_hf = time.time()
auto_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens_hf = auto_tokenizer.tokenize(text)
end_hf = time.time()

print("\n[AutoTokenizer]")
print("Token count:", len(tokens_hf))
print("Time taken:", round(end_hf - start_hf, 6), "seconds")

# -------------------- SentencePiece --------------------
start_sp = time.time()
sp = spm.SentencePieceProcessor(model_file="spiece.model")
tokens_sp = sp.encode(text, out_type=str)
end_sp = time.time()

print("\n[SentencePiece]")
print("Token count:", len(tokens_sp))
print(tokens_sp)
print("Time taken:", round(end_sp - start_sp, 6), "seconds")

# -------------------- spaCy --------------------
start_spacy = time.time()
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
tokens_spacy = [token.text for token in doc]
end_spacy = time.time()

print("\n[spaCy]")
print("Token count:", len(tokens_spacy))
print("Time taken:", round(end_spacy - start_spacy, 6), "seconds")

# -------------------- NLTK (Treebank tokenizer) --------------------
start_nltk = time.time()
tokens_nltk = TreebankWordTokenizer().tokenize(text)
end_nltk = time.time()

print("\n[NLTK - TreebankWordTokenizer]")
print("Token count:", len(tokens_nltk))
print("Time taken:", round(end_nltk - start_nltk, 6), "seconds")

# -------------------- tokenizers (Rust backend) --------------------
start_tok = time.time()
fast_tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
output = fast_tokenizer.encode(text)
tokens_fast = output.tokens
end_tok = time.time()

