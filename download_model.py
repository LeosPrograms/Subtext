"""Download GPT-2 model weights to ./model/gpt2 for offline / bundled use."""

import os
from transformers import AutoModelForCausalLM, AutoTokenizer

DEST = os.path.join(os.path.dirname(__file__), 'model', 'gpt2')
MODEL = 'gpt2'

print(f'Downloading {MODEL} -> {DEST}')
AutoTokenizer.from_pretrained(MODEL).save_pretrained(DEST)
AutoModelForCausalLM.from_pretrained(MODEL).save_pretrained(DEST)
print('Done.')
