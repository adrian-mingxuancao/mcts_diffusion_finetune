"""
Proteina Expert Model Integration

Integration for Proteina expert model for motif scaffolding via denovo-protein-server.
"""

import requests
import json
import time
from typing import Dict, Optional, List
import numpy as np

class ProteineaExpertModel:
    """Proteina expert model for motif scaffolding via denovo-protein-server."""
    
    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url
        self.name = "Proteina"
        self.timeout = 30  # seconds
    
    def get_name(self) -> str:
        """Get model name for compatibility with test framework."""
        return self.name
    
    def _check_server_health(self) -> bool:
        """Check if Proteina server is running."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_scaffold(self, motif_data, scaffold_length: int, **kwargs) -> Optional[Dict]:
        """Generate scaffold using Proteina via HTTP API."""
        try:
            print(f"   🔄 {self.name} generating scaffold via denovo-protein-server...")
            
            # Check server health
            if not self._check_server_health():
                print(f"   ⚠️ {self.name} server not available, using fallback")
                return self._fallback_generation(motif_data, scaffold_length)
            
            # Prepare request for Proteina
            target_length = len(motif_data.full_sequence)
            motif_length = len(motif_data.motif_sequence)
            actual_scaffold_length = target_length - motif_length
            
            # Proteina generates protein backbones and sequences
            request_data = {
                "length": target_length,
                "num_samples": 1,
                "generate_sequences": True,
                "num_seq_per_target": 1,
                "temperature": kwargs.get('temperature', 1.0),
                "flow_matching_steps": 50,  # Reasonable for motif scaffolding
                "noise_scale": 0.1
            }
            
            # Make API request
            response = requests.post(
                f"{self.server_url}/generate",
                json=request_data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ {self.name} API response received")
                
                # Extract generated sequence
                if 'sequences' in result and result['sequences']:
                    generated_seq = result['sequences'][0]
                    
                    # Ensure motif is preserved
                    motif_preserved = motif_data.motif_sequence in generated_seq
                    
                    print(f"   ✅ {self.name} generated: {len(generated_seq)} residues")
                    print(f"   🎯 Motif preserved: {motif_preserved}")
                    
                    return {
                        'full_sequence': generated_seq,
                        'motif_preserved': motif_preserved,
                        'scaffold_length': len(generated_seq) - len(motif_data.motif_sequence),
                        'method': 'proteina_api',
                        'motif_sequence': motif_data.motif_sequence,
                        'temperature': kwargs.get('temperature', 1.0),
                        'structure_sequence': result.get('structures', [''])[0] if 'structures' in result else ''
                    }
                else:
                    print(f"   ⚠️ {self.name} server returned no sequences: {list(result.keys())}")
                    return self._fallback_generation(motif_data, scaffold_length)
            else:
                print(f"   ⚠️ {self.name} server error: {response.status_code} - {response.text}")
                return self._fallback_generation(motif_data, scaffold_length)
                
        except Exception as e:
            print(f"   ⚠️ {self.name} generation failed: {e}")
            return self._fallback_generation(motif_data, scaffold_length)
    
    def _fallback_generation(self, motif_data, scaffold_length: int) -> Dict:
        """Fallback generation when server is not available."""
        import random
        
        # Generate Proteina-biased scaffold (more diverse sequences)
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        
        # Calculate actual scaffold length
        target_length = len(motif_data.full_sequence)
        actual_scaffold_length = target_length - len(motif_data.motif_sequence)
        
        # Insert motif at random position
        motif_pos = random.randint(0, max(0, actual_scaffold_length - len(motif_data.motif_sequence)))
        
        # Generate diverse scaffold
        left_scaffold = ''.join(random.choices(amino_acids, k=motif_pos))
        right_scaffold = ''.join(random.choices(amino_acids, k=actual_scaffold_length - motif_pos))
        
        full_sequence = left_scaffold + motif_data.motif_sequence + right_scaffold
        
        return {
            'full_sequence': full_sequence,
            'motif_preserved': motif_data.motif_sequence in full_sequence,
            'scaffold_length': actual_scaffold_length,
            'method': 'proteina_fallback'
        }
    
    def get_name(self) -> str:
        return self.name