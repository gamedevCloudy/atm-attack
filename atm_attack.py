import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline # Using latent diffusion
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
from torch.distributions import Gumbel
import numpy as np
from typing import List, Dict, Tuple
import bert_score

class GumbelSoftmaxSampler(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
    def forward(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Apply Gumbel-Softmax sampling to logits
        Args:
            logits: Shape (batch_size, sequence_length, vocab_size)
            temperature: Softmax temperature
        Returns:
            Sampled embeddings
        """
        gumbel = Gumbel(0, 1).sample(logits.shape).to(logits.device)
        y = logits + gumbel
        return F.softmax(y / temperature, dim=-1)

class ATMAttack:
    def __init__(
        self,
        sd_model_path: str = "OFA-Sys/small-stable-diffusion-v0",
        margin: float = 0.5,
        temperature: float = 0.1,
        fluency_weight: float = 0.3,
        similarity_weight: float = 0.3
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        # Replace Stable Diffusion with a Latent Diffusion Model
        self.sd_pipeline = StableDiffusionPipeline.from_pretrained(sd_model_path).to(self.device)
        # Replace CLIP with a distilled version
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        
        # Replace GPT-2 with DistilGPT2
        self.lm_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.lm_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(self.device)
        
        # Initialize Gumbel-Softmax sampler
        self.gumbel_sampler = GumbelSoftmaxSampler(
            self.lm_tokenizer.vocab_size,
            self.clip_model.config.text_config.hidden_size
        ).to(self.device)
        
        # Parameters
        self.margin = margin
        self.temperature = temperature
        self.fluency_weight = fluency_weight
        self.similarity_weight = similarity_weight

    def compute_margin_loss(
        self,
        logits: torch.Tensor,
        target_class: int
    ) -> torch.Tensor:
        """Compute margin loss for classification"""
        target_logits = logits[:, target_class]
        other_logits = torch.cat([logits[:, :target_class], logits[:, target_class+1:]], dim=1)
        max_other = torch.max(other_logits, dim=1)[0]
        return torch.clamp(target_logits - max_other + self.margin, min=0)

    def compute_fluency_loss(
        self,
        sampled_tokens: torch.Tensor,
        original_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Compute language model fluency loss"""
        outputs = self.lm_model(input_ids=original_tokens)
        lm_logits = outputs.logits[:, :-1, :]
        sampled_tokens = sampled_tokens[:, 1:, :]
        return F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), 
                             sampled_tokens.view(-1, sampled_tokens.size(-1)))

    def compute_similarity_loss(
        self,
        original_text: str,
        modified_text: str
    ) -> torch.Tensor:
        """Compute BERTScore-based similarity loss"""
        # Replace BERTScore with DistilBERT
        P, R, F1 = bert_score.score(
            [modified_text], 
            [original_text], 
            model_type='distilbert-base-uncased', 
            lang='en', 
            verbose=False
        )
        return 1 - F1.mean()

    def generate_adversarial_prompt(
        self,
        clean_prompt: str,
        target_class: int,
        num_iterations: int = 100,
        num_candidates: int = 50,
        learning_rate: float = 0.01
    ) -> List[str]:
        """
        Generate adversarial prompts using the ATM method
        
        Args:
            clean_prompt: Original prompt
            target_class: Desired class index
            num_iterations: Number of search iterations
            num_candidates: Number of attack candidates
            learning_rate: Learning rate for optimization
            
        Returns:
            List of successful adversarial prompts
        """
        # Tokenize clean prompt
        tokens = self.lm_tokenizer(clean_prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(self.device)
        
        # Initialize logits for Gumbel-Softmax
        logits = torch.randn(
            (1, input_ids.size(1), self.lm_tokenizer.vocab_size),
            requires_grad=True,
            device=self.device
        )
        
        optimizer = torch.optim.Adam([logits], lr=learning_rate)
        
        # Search stage
        for _ in range(num_iterations):
            optimizer.zero_grad()
            
            # Sample tokens using Gumbel-Softmax
            sampled_tokens = self.gumbel_sampler(logits, self.temperature)
            
            # Generate image using modified prompt
            modified_text = self.lm_tokenizer.decode(
                torch.argmax(sampled_tokens[0], dim=-1)
            )
            image = self.sd_pipeline(modified_text).images[0]
            
            # Get CLIP features
            image_features = self.clip_model.get_image_features(
                self.clip_processor(images=image, return_tensors="pt").pixel_values.to(self.device)
            )
            
            # Compute losses
            margin_loss = self.compute_margin_loss(image_features, target_class)
            fluency_loss = self.compute_fluency_loss(sampled_tokens, input_ids)
            similarity_loss = self.compute_similarity_loss(clean_prompt, modified_text)
            
            # Total loss
            loss = (margin_loss + 
                   self.fluency_weight * fluency_loss +
                   self.similarity_weight * similarity_loss)
            
            loss.backward()
            optimizer.step()
        
        # Attack stage
        successful_prompts = []
        with torch.no_grad():
            for _ in range(num_candidates):
                sampled_tokens = self.gumbel_sampler(logits, self.temperature)
                modified_text = self.lm_tokenizer.decode(
                    torch.argmax(sampled_tokens[0], dim=-1)
                )
                
                # Generate and classify image
                image = self.sd_pipeline(modified_text).images[0]
                image_features = self.clip_model.get_image_features(
                    self.clip_processor(images=image, return_tensors="pt").pixel_values.to(self.device)
                )
                
                predicted_class = torch.argmax(image_features[0]).item()
                
                if predicted_class != target_class:
                    successful_prompts.append(modified_text)
        
        return successful_prompts
