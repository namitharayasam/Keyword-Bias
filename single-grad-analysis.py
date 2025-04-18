//analysis for single text image pair 

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import matplotlib.pyplot as plt

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# Load image
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

# Question
# question = "What is the color of the sky"
question = "color of sky?"
# question = 'color'

# Tokenize
inputs = processor(text=[question], images=image, return_tensors="pt", padding=True)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"].float()

# Get image features
with torch.no_grad():
    pixel_values = inputs["pixel_values"]
    image_output = model.vision_model(pixel_values=pixel_values).pooler_output
    image_output = image_output / image_output.norm(p=2, dim=-1, keepdim=True)

# Get token embeddings and enable gradients
text_embeds = model.text_model.embeddings.token_embedding(input_ids)
text_embeds.requires_grad_()
text_embeds.retain_grad()

# Forward through the text encoder using inputs_embeds
encoder_outputs = model.text_model.encoder(
    inputs_embeds=text_embeds,
    attention_mask=attention_mask,
    causal_attention_mask=None
)
hidden_states = encoder_outputs.last_hidden_state

# Pooling - use [CLS] token
text_output = hidden_states[:, 0, :]
text_output = text_output / text_output.norm(p=2, dim=-1, keepdim=True)

# Cosine similarity
# similarity = (text_output @ image_output.T).squeeze()
# Project to CLIP's shared embedding space
text_emb = model.text_projection(text_output)         # shape: (1, 512)
image_emb = model.visual_projection(image_output)     # shape: (1, 512)

# Normalize
text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

# Cosine similarity
similarity = (text_emb @ image_emb.T).squeeze()


# Backpropagation
similarity.backward()

# Extract token-wise gradient importance
token_grads = text_embeds.grad.norm(dim=-1).squeeze().detach()
tokens = processor.tokenizer.convert_ids_to_tokens(input_ids[0])

# Filter out special tokens and plot
filtered_tokens, filtered_grads = [], []
for token, grad, attn in zip(tokens, token_grads, attention_mask[0]):
    if attn == 1 and token not in processor.tokenizer.all_special_tokens:
        filtered_tokens.append(token)
        filtered_grads.append(grad.item())

# Plot
plt.figure(figsize=(12, 2))
plt.bar(range(len(filtered_grads)), filtered_grads, tick_label=filtered_tokens, color='skyblue')
plt.ylabel("Gradient Magnitude")
plt.title(f"Word Importance for: \"{question}\"")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
