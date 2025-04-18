import torch
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torchvision import transforms
import os
from tqdm import tqdm
import random

# Create output directory for plots
os.makedirs("gradient_plots", exist_ok=True)

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# Load FairFace dataset
fairface = load_dataset("HuggingFaceM4/FairFace", "0.25", split="train", trust_remote_code=True)
fairface_subset = fairface.select(range(300))

# Define mappings for FairFace integer attributes
GENDER_MAP = {
    0: "female",
    1: "male"
}

AGE_MAP = {
    0: "0-2 years old",
    1: "3-9 years old",
    2: "10-19 years old",
    3: "20-29 years old",
    4: "30-39 years old",
    5: "40-49 years old",
    6: "50-59 years old",
    7: "60-69 years old",
    8: "70+ years old"
}

RACE_MAP = {
    0: "White",
    1: "Black",
    2: "Latino Hispanic",
    3: "East Asian",
    4: "Southeast Asian",
    5: "Indian",
    6: "Middle Eastern"
}

# Define a function to generate queries based on image attributes
def generate_query(image_info):
    """Generate a natural language query based on image attributes."""
    attributes = []
    
    # Use available attributes from the dataset with proper mapping
    if 'gender' in image_info:
        gender_val = image_info['gender']
        gender_str = GENDER_MAP.get(gender_val, f"person of gender {gender_val}")
        attributes.append(f"a {gender_str}")
    
    if 'age' in image_info:
        age_val = image_info['age']
        age_str = AGE_MAP.get(age_val, f"person of age group {age_val}")
        attributes.append(f"who appears {age_str}")
    
    if 'race' in image_info:
        race_val = image_info['race']
        race_str = RACE_MAP.get(race_val, f"of ethnicity type {race_val}")
        attributes.append(f"with {race_str} ethnicity")
    
    # If no attributes could be extracted, use a generic description
    if not attributes:
        return "Is this a person in the image?"
    
    # Create different query types
    query_types = [
        f"Is this {' '.join(attributes)}?",
        f"Can you see {' '.join(attributes)}?",
        f"The image shows {' '.join(attributes)}.",
        f"A photo of {' '.join(attributes)}."
    ]
    
    return random.choice(query_types)

# Function to analyze word contribution using gradients
def analyze_word_contribution(image, query, model, processor):
    """Analyze word contribution for an image-query pair using gradients."""
    # Tokenize
    inputs = processor(text=[query], images=image, return_tensors="pt", padding=True)
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
    
    # Filter out special tokens
    filtered_tokens, filtered_grads = [], []
    for token, grad, attn in zip(tokens, token_grads, attention_mask[0]):
        if attn == 1 and token not in processor.tokenizer.all_special_tokens:
            filtered_tokens.append(token)
            filtered_grads.append(grad.item())
    
    return {
        'tokens': filtered_tokens,
        'gradients': filtered_grads,
        'similarity': similarity.item()
    }

# Function to plot and save word contribution analysis
def plot_word_contribution(analysis_result, query, sample_idx):
    """Plot and save word contribution visualization."""
    plt.figure(figsize=(12, 3))
    plt.bar(range(len(analysis_result['gradients'])), 
            analysis_result['gradients'], 
            tick_label=analysis_result['tokens'], 
            color='skyblue')
    plt.ylabel("Gradient Magnitude")
    plt.title(f"Sample {sample_idx}: Word Importance for \"{query}\" (Sim: {analysis_result['similarity']:.4f})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"gradient_plots/sample_{sample_idx}.png")
    plt.close()

# Initialize lists to store results
all_results = []

# Define a fixed query if needed
FIXED_QUERY = "Is this a person in the image?"

# Create a counter for successfully processed samples
success_count = 0

# Process the subset of samples
for idx, sample in enumerate(tqdm(fairface_subset, desc="Processing samples")):
    try:
        # Generate query for this image (or use fixed query)
        # Uncomment the line below to use dynamic queries based on attributes
        query = generate_query(sample)
        # Uncomment the line below to use the same query for all images
        # query = FIXED_QUERY
        
        # Get the image
        image = sample['image']
        
        # Analyze word contribution
        analysis = analyze_word_contribution(image, query, model, processor)
        
        # Add sample info to the analysis
        analysis['sample_idx'] = idx
        analysis['query'] = query
        analysis['gender'] = GENDER_MAP.get(sample.get('gender', -1), 'Unknown')
        analysis['age'] = AGE_MAP.get(sample.get('age', -1), 'Unknown')
        analysis['race'] = RACE_MAP.get(sample.get('race', -1), 'Unknown')
        
        # Store results
        all_results.append(analysis)
        
        # Plot and save
        plot_word_contribution(analysis, query, idx)
        
        # Increment success counter
        success_count += 1
        
        # Print periodic status
        if success_count % 10 == 0:
            print(f"Successfully processed {success_count} samples")
        
        # Free memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"Error processing sample {idx}: {e}")

print(f"\nTotal successful samples: {success_count} out of {len(fairface_subset)}")

# Convert results to DataFrame for analysis
def convert_to_df(all_results):
    """Convert analysis results to DataFrame format."""
    rows = []
    for result in all_results:
        for token, gradient in zip(result['tokens'], result['gradients']):
            rows.append({
                'sample_idx': result['sample_idx'],
                'query': result['query'],
                'token': token,
                'gradient': gradient,
                'similarity': result['similarity'],
                'gender': result['gender'],
                'age': result['age'],
                'race': result['race']
            })
    return pd.DataFrame(rows)

# Check if we have any results before proceeding
if not all_results:
    print("\nNo successful analyses completed. Cannot create results dataframe.")
    # Create empty dataframe with proper columns for saving
    results_df = pd.DataFrame(columns=[
        'sample_idx', 'query', 'token', 'gradient', 'similarity', 
        'gender', 'age', 'race'
    ])
else:
    results_df = convert_to_df(all_results)
    
    # Save results to CSV
    results_df.to_csv("fairface_gradient_analysis.csv", index=False)
    
    # Perform some basic analysis
    if len(results_df) > 0:
        # Top words by average gradient magnitude
        top_words = results_df.groupby('token')['gradient'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        
        # Only filter if we have enough data
        min_count = min(5, max(1, len(all_results) // 10))
        top_words_filtered = top_words[top_words['count'] >= min_count]
        
        print("\nTop 20 Words by Average Gradient Magnitude:")
        print(top_words_filtered.head(20))
        
        # Analysis by demographic attributes (if we have enough data)
        if 'gender' in results_df.columns and len(results_df['gender'].unique()) > 1:
            print("\nAverage Gradient by Gender:")
            gender_analysis = results_df.groupby(['gender', 'token'])['gradient'].mean().reset_index()
            for gender in gender_analysis['gender'].unique():
                gender_tokens = gender_analysis[gender_analysis['gender'] == gender]
                if len(gender_tokens) > 0:
                    top_tokens = gender_tokens.sort_values('gradient', ascending=False).head(10)
                    print(f"\n{gender}:")
                    print(top_tokens)
        
        if 'race' in results_df.columns and len(results_df['race'].unique()) > 1:
            print("\nAverage Gradient by Race:")
            race_analysis = results_df.groupby(['race', 'token'])['gradient'].mean().reset_index()
            for race in race_analysis['race'].unique():
                race_tokens = race_analysis[race_analysis['race'] == race]
                if len(race_tokens) > 0:
                    top_tokens = race_tokens.sort_values('gradient', ascending=False).head(5)
                    print(f"\n{race}:")
                    print(top_tokens)
    else:
        print("No data available for analysis.")

# Create summary visualization if there's data
if len(all_results) > 0 and 'top_words' in locals():
    try:
        plt.figure(figsize=(15, 8))
        # Take at most 20 words, but could be fewer
        top_n_words = min(20, len(top_words))
        top_n_words_list = top_words.head(top_n_words).index.tolist()
        
        if top_n_words_list:
            subset = results_df[results_df['token'].isin(top_n_words_list)]
            if not subset.empty:
                subset_pivot = subset.pivot_table(index='token', values='gradient', aggfunc='mean')
                subset_pivot.sort_values('gradient', ascending=False).plot(kind='bar', color='cornflowerblue')
                plt.title(f"Average Gradient Magnitude for Top {top_n_words} Words")
                plt.ylabel("Average Gradient Magnitude")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig("gradient_plots/top_words_summary.png")
                print("\nSummary visualization created and saved to 'gradient_plots/top_words_summary.png'")
            else:
                print("Could not create summary visualization - subset data is empty")
        else:
            print("No top words available for visualization")
    except Exception as e:
        print(f"Error creating summary visualization: {e}")
else:
    print("Not enough data to create summary visualization")

# Alternative approach for a limited dataset: generate a hardcoded query for all images
print("\nIf you're still having issues with the data, try using a fixed query for all images.")
print("Modify the code to use FIXED_QUERY = 'Is this a person?' for all samples instead of dynamic queries.")

print("\nAnalysis complete!")
