import os

import json
import tempfile
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model,save_model

from src.models.utils import top_k_top_p_filtering
from src.models.vision_transformer import ViT
from src.models.language_model import LlamaTransformer
from src.models.modality_projector import ModalityProjector

from src.models.config import GPTConfig,SigLIPConfig
from src.data.processors import get_tokenizer

class VisionLanguageModel(nn.Module):
    def __init__(self,cfg:SigLIPConfig,load_backbone=True):
        super().__init__()
        
        self.cfg=cfg
        
        if load_backbone:
            print("Load from backbone weights")
            self.vision_encoder= ViT.from_pretrained(cfg)
        else:
            self.vision_encoder=ViT(cfg)
            self.decoder=LlamaTransformer(cfg)
        self.MP=ModalityProjector(cfg)
        
        self.load_backbone=load_backbone
        self.tokenizer=get_tokenizer(cfg.lm_tokenizer,cfg.vlm_extra_tokens,cfg.vlm_chat_template)
        
    def _replace_img_tokens_with_embd(self,input_ids:torch.Tensor,token_embd:torch.Tensor,image_embd:torch.Tensor):
        """
        Replace every image-token placeholder in `input_ids` with the corresponding slice
        from `image_embd`. Supports an arbitrary number of image-token placeholders per sample.
        The first example in the batch might have 2 images and the second none.
        """
        
        # Clone the original embedding to avoid in-plcae issues
        updata_token_embd=token_embd.clone()
        
        # Build a mask of all image-token posintions: shape [B,seq_len]
        mask=(input_ids==self.tokenizer.image_token_id)
        updata_token_embd[mask]=image_embd.view(-1,image_embd.size(-1)).to(updata_token_embd.dtype)
        
        return updata_token_embd
    
    
    def _process_images(self,images,device):
        if isinstance(images,list):
            if images and isinstance(images[0],list):
                images=[img for sublist in images for img in sublist]
        
            if not images: # Handle cases with no images
                return None
            else:
                return torch.cat(images,dim=0).to(device)
        
        return images # Already a tensor
    
    def forward(self,input_ids: torch.Tensor,images,attention_mask=None,targets=None):
        images_tensor=self._process_images(images,input_ids.device)
        token_embd=self.decoder.token_embedding(input_ids)
        
        if images_tensor is not None:
            image_embd=self.vision_encoder(images_tensor)
            image_embd=self.MP(image_embd) # [n_images,mp_image_token_length,n_embd]
            
            token_embd=self._replace_img_tokens_with_embd(input_ids,token_embd,image_embd)
            
        logits,_=self.decoder(token_embd,attention_mask=attention_mask)
        
        loss=None
        if targets is not None:
            logits=self.decoder.head(logits) # Apply LM head
            # Loss is calculated over all tokens, but `targets` (labels) will have -100 for non-answer tokens.
            # No need to slice logits based on image embedding size here, as the target mask handles it.
            loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1),ignore_index=-100)
        
        return logits,loss
        
    
    @torch.inference_mode()
    def generate(self,input_ids: torch.Tensor,images,attention_mask=None,max_new_tokens:int=5,top_k:int=50,top_p:float=0.9,temperature:float=0.5,greedy=False):
        image_tensor=self._process_images(images,input_ids.device)
        token_embd=self.decoder.token_embedding(input_ids)
        
        if image_tensor is not None:
            # 1. Process image if present
            image_embd=self.vision_encoder(image_tensor) # [bsz,image_feat_len,n_embd]
            image_embd=self.MP(image_embd) # [bsz,mp_image_token_lenght,n_embd]
            
            # 2. Combine image and text emebeddings
            token_embd=self._replace_img_tokens_with_embd(input_ids,token_embd,image_embd)
            
        current_total_seq_len=token_embd.size(1)
        bsz=input_ids.size(0) # or token_embd.size(0)
        
        # --- Multimodal Prefill Pashe ---
        prefill_output,kv_cache_list=self.decoder(
            token_embd,
            attention_mask=attention_mask,
            kv_cache=None,
            start_pos=0
        )
        
        last_token_output_from_prefill=prefill_output[:,-1,:]
        
        if not self.decoder.lm_use_tokens:
            current_logits=self.decoder.head(last_token_output_from_prefill)
        else:
            current_logits=last_token_output_from_prefill
            
        # Store newly generated token IDs
        newly_generated_ids_list=[]
        
        # --- Decade Phase by sampling tokens autoregressively using the kv-cache ---
        for _ in range(max_new_tokens):
            if greedy:
                next_token_id=torch.argmax(current_logits,dim=-1,keepdim=True)
            else:
                fitered_logits=top_k_top_p_filtering(current_logits,top_k=top_k,top_p=top_p)
                probs=torch.softmax(fitered_logits/temperature,dim=-1)
                next_token_id=torch.multinomial(probs,num_samples=1)
            
            newly_generated_ids_list.append(next_token_id)
            
            # Embd the newly generated token
            next_token_embd=self.decoder.token_embedding(next_token_id) # [bsz,1,n_embd]
            
            # The start_pos for the new token is the current total sequence length *before* adding this new token
            current_token_start_pos=current_total_seq_len
            current_total_seq_len+=1
            
            # update attention mask
            if attention_mask is not None:
                attention_mask=torch.cat((attention_mask, torch.ones((bsz, 1), device=attention_mask.device, dtype=attention_mask.dtype)), dim=1)
            
            # With KV cache: only precess the new token
            decode_step_output,kv_cache_list=self.decoder(
                next_token_embd,
                attention_mask=attention_mask,
                kv_chache=kv_cache_list,
                start_pos=current_token_start_pos
            )
            last_token_output = decode_step_output[:, -1, :] 
            
            # Apply head to get logits (if model is in embedding mode)
            if not self.decoder.lm_use_tokens:
                current_logits = self.decoder.head(last_token_output)
            else:
                current_logits = last_token_output
        
        if not newly_generated_ids_list: # Handle case where max_new_tokens might be 0
            return torch.empty((bsz,0), dtype=torch.long, device=input_ids.device)

        generated_ids = torch.cat(newly_generated_ids_list, dim=1)

        # Post-process to handle EOS token.
        if self.tokenizer.eos_token_id is not None and generated_ids.numel() > 0: # Ensure generated_ids is not empty
            seq_len = generated_ids.size(1)
            device = generated_ids.device

            eos_mask = (generated_ids == self.tokenizer.eos_token_id) # Create a boolean mask for EOS tokens

            col_indices_for_min = torch.arange(seq_len, device=device) # Create column indices [0, 1, ..., seq_len-1]
            
            # In eos_mask, mark positions with actual col_idx, others with a large number
            masked_col_indices = torch.where(eos_mask, col_indices_for_min.unsqueeze(0).expand_as(generated_ids), seq_len + 1) 

            first_eos_indices_values = torch.min(masked_col_indices, dim=1).values
            
            # Clamp values to seq_len (if no EOS found, min will be seq_len + 1, clamp brings it to seq_len0. This means if no EOS, or EOS is the last token, no replacement will happen for that sample.
            actual_first_eos_indices = torch.clamp(first_eos_indices_values, max=seq_len)

            # Create column indices for comparison, shape [bsz, seq_len]
            col_indices_for_comparison = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(generated_ids)
            
            # Tokens are replaced if their column index is greater than the index of the first EOS token
            replace_mask = col_indices_for_comparison > actual_first_eos_indices.unsqueeze(1)
            
            generated_ids[replace_mask] = self.tokenizer.eos_token_id
        
        return generated_ids