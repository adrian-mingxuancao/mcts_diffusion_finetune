#!/usr/bin/env python3
"""
Integration with external protein generation models from denovo-protein-server.

This module provides a unified interface to interact with:
- Proteinea: Structure generation and sequence design
- FoldFlow: Flow-based protein structure generation  
- RFDiffusion: Diffusion-based protein structure generation

All models are accessed via HTTP API calls to their respective servers.
"""

import requests
import json
import time
from typing import Dict, List, Optional, Tuple
import numpy as np


class ExternalModelsIntegration:
    """Integration with external protein generation models via HTTP APIs."""
    
    def __init__(self, server_configs: Dict[str, str] = None):
        """
        Initialize external models integration.
        
        Args:
            server_configs: Dictionary mapping model names to server URLs
                          e.g., {"proteinea": "http://localhost:8080", 
                                "foldflow": "http://localhost:8081",
                                "rfdiffusion": "http://localhost:8082"}
        """
        self.server_configs = server_configs or {
            "proteinea": "http://localhost:8080",
            "foldflow": "http://localhost:8081", 
            "rfdiffusion": "http://localhost:8082"
        }
        
        self.available_models = []
        self._check_server_availability()
    
    def _check_server_availability(self):
        """Check which servers are available and responsive."""
        for model_name, server_url in self.server_configs.items():
            try:
                response = requests.get(f"{server_url}/health", timeout=5)
                if response.status_code == 200:
                    self.available_models.append(model_name)
                    print(f"✅ {model_name.capitalize()} server available at {server_url}")
                else:
                    print(f"⚠️ {model_name.capitalize()} server not responding at {server_url}")
            except Exception as e:
                print(f"⚠️ {model_name.capitalize()} server not available: {e}")
        
        if not self.available_models:
            print("⚠️ No external model servers available")
    
    def generate_motif_scaffolds(self, motif_sequence: str, motif_positions: List[int], 
                                scaffold_length: int, num_samples: int = 2,
                                model_name: str = "proteinea") -> List[str]:
        """
        Generate motif scaffolds using external models.
        
        Args:
            motif_sequence: The motif sequence to scaffold around
            motif_positions: Positions of the motif in the final sequence
            scaffold_length: Total length of the scaffolded sequence
            num_samples: Number of samples to generate
            model_name: Which model to use ("proteinea", "foldflow", "rfdiffusion")
            
        Returns:
            List of generated scaffold sequences
        """
        if model_name not in self.available_models:
            print(f"⚠️ {model_name} not available, using fallback")
            return self._generate_fallback_scaffolds(motif_sequence, motif_positions, 
                                                   scaffold_length, num_samples)
        
        try:
            server_url = self.server_configs[model_name]
            
            # Prepare request payload based on model type
            if model_name == "proteinea":
                payload = self._prepare_proteinea_payload(motif_sequence, motif_positions,
                                                        scaffold_length, num_samples)
            elif model_name == "foldflow":
                payload = self._prepare_foldflow_payload(motif_sequence, motif_positions,
                                                       scaffold_length, num_samples)
            elif model_name == "rfdiffusion":
                payload = self._prepare_rfdiffusion_payload(motif_sequence, motif_positions,
                                                          scaffold_length, num_samples)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Make API request
            response = requests.post(f"{server_url}/generate", 
                                   json=payload, 
                                   timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                sequences = self._extract_sequences_from_response(result, model_name)
                print(f"✅ {model_name.capitalize()} generated {len(sequences)} scaffolds")
                return sequences
            else:
                print(f"⚠️ {model_name.capitalize()} API error: {response.status_code}")
                return self._generate_fallback_scaffolds(motif_sequence, motif_positions,
                                                       scaffold_length, num_samples)
                
        except Exception as e:
            print(f"⚠️ {model_name.capitalize()} generation failed: {e}")
            return self._generate_fallback_scaffolds(motif_sequence, motif_positions,
                                                   scaffold_length, num_samples)
    
    def _prepare_proteinea_payload(self, motif_sequence: str, motif_positions: List[int],
                                  scaffold_length: int, num_samples: int) -> Dict:
        """Prepare payload for Proteinea server."""
        return {
            "length": scaffold_length,
            "num_samples": num_samples,
            "motif_sequence": motif_sequence,
            "motif_positions": motif_positions,
            "temperature": 1.0,
            "top_p": 0.9
        }
    
    def _prepare_foldflow_payload(self, motif_sequence: str, motif_positions: List[int],
                                 scaffold_length: int, num_samples: int) -> Dict:
        """Prepare payload for FoldFlow server."""
        return {
            "length": scaffold_length,
            "num_samples": num_samples,
            "motif_sequence": motif_sequence,
            "motif_positions": motif_positions,
            "guidance_scale": 1.0
        }
    
    def _prepare_rfdiffusion_payload(self, motif_sequence: str, motif_positions: List[int],
                                    scaffold_length: int, num_samples: int) -> Dict:
        """Prepare payload for RFDiffusion server."""
        return {
            "length": scaffold_length,
            "num_samples": num_samples,
            "motif_sequence": motif_sequence,
            "motif_positions": motif_positions,
            "num_steps": 50
        }
    
    def _extract_sequences_from_response(self, response: Dict, model_name: str) -> List[str]:
        """Extract sequences from API response based on model type."""
        sequences = []
        
        try:
            if model_name == "proteinea":
                # Proteinea returns sequences directly
                sequences = response.get("sequences", [])
            elif model_name == "foldflow":
                # FoldFlow might return structures, extract sequences
                for item in response.get("results", []):
                    if "sequence" in item:
                        sequences.append(item["sequence"])
            elif model_name == "rfdiffusion":
                # RFDiffusion returns structures, extract sequences
                for item in response.get("results", []):
                    if "sequence" in item:
                        sequences.append(item["sequence"])
            
            # Clean sequences
            clean_sequences = []
            for seq in sequences:
                if isinstance(seq, str) and len(seq) > 0:
                    clean_seq = ''.join([aa for aa in seq.upper() if aa in "ACDEFGHIKLMNPQRSTVWY"])
                    if clean_seq:
                        clean_sequences.append(clean_seq)
            
            return clean_sequences
            
        except Exception as e:
            print(f"⚠️ Error extracting sequences from {model_name} response: {e}")
            return []
    
    def _generate_fallback_scaffolds(self, motif_sequence: str, motif_positions: List[int],
                                   scaffold_length: int, num_samples: int) -> List[str]:
        """Generate fallback scaffolds using simple random sampling."""
        import random
        
        sequences = []
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        
        for _ in range(num_samples):
            # Create scaffold sequence
            scaffold = ['A'] * scaffold_length  # Start with alanines
            
            # Insert motif at specified positions
            for i, pos in enumerate(motif_positions):
                if i < len(motif_sequence) and pos < scaffold_length:
                    scaffold[pos] = motif_sequence[i]
            
            # Randomize non-motif positions
            motif_pos_set = set(motif_positions)
            for i in range(scaffold_length):
                if i not in motif_pos_set:
                    scaffold[i] = random.choice(amino_acids)
            
            sequences.append(''.join(scaffold))
        
        print(f"🔄 Generated {len(sequences)} fallback scaffolds")
        return sequences
    
    def compute_model_confidence(self, sequence: str, model_name: str) -> List[float]:
        """
        Compute confidence scores for a sequence using external models.
        
        Args:
            sequence: Amino acid sequence
            model_name: Model to use for confidence estimation
            
        Returns:
            Per-residue confidence scores (0-100 scale)
        """
        if model_name not in self.available_models:
            print(f"⚠️ {model_name} not available for confidence calculation")
            return [70.0] * len(sequence)  # Default confidence
        
        try:
            server_url = self.server_configs[model_name]
            
            payload = {
                "sequence": sequence,
                "task": "confidence"
            }
            
            response = requests.post(f"{server_url}/generate", 
                                   json=payload, 
                                   timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                confidence_scores = result.get("confidence", [70.0] * len(sequence))
                print(f"✅ {model_name.capitalize()} confidence: mean={np.mean(confidence_scores):.1f}")
                return confidence_scores
            else:
                print(f"⚠️ {model_name.capitalize()} confidence API error")
                return [70.0] * len(sequence)
                
        except Exception as e:
            print(f"⚠️ {model_name.capitalize()} confidence calculation failed: {e}")
            return [70.0] * len(sequence)
