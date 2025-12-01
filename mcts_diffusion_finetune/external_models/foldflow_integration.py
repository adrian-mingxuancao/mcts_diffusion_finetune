"""
FlowFlow Expert Model Integration

Integration for FlowFlow expert model for motif scaffolding via denovo-protein-server.
"""

import requests
import json
import time
from typing import Dict, Optional, List
import numpy as np

class FlowFlowExpertModel:
    """FlowFlow expert model for motif scaffolding via denovo-protein-server."""
    
    def __init__(self, server_url: str = "http://localhost:8081"):
        self.server_url = server_url
        self.name = "FlowFlow"
        self.timeout = 30  # seconds
    
    def _check_server_health(self) -> bool:
        """Check if FlowFlow server is running."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_scaffold(self, motif_data, scaffold_length: int, **kwargs) -> Optional[Dict]:
        """Generate scaffold using FlowFlow via HTTP API."""
        try:
            print(f"   🔄 {self.name} generating scaffold via denovo-protein-server...")
            
            # Check server health
            if not self._check_server_health():
                print(f"   ⚠️ {self.name} server not available, using fallback")
                return self._fallback_generation(motif_data, scaffold_length)
            
            # Prepare request for FlowFlow
            target_length = len(motif_data.full_sequence)
            motif_length = len(motif_data.motif_sequence)
            actual_scaffold_length = target_length - motif_length
            
            # FlowFlow generates protein backbones, so we need to provide structure context
            request_data = {
                "length": target_length,
                "num_samples": 1,
                "num_t": 50,  # Flow matching steps
                "temperature": kwargs.get('temperature', 1.0),
                "motif_sequence": motif_data.motif_sequence,
                "motif_positions": motif_data.motif_positions[:10] if len(motif_data.motif_positions) > 10 else motif_data.motif_positions,
                "conditional": True  # Use motif conditioning
            }
            
            # Make API request
            response = requests.post(
                f"{self.server_url}/generate",
                json=request_data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract generated sequence
                if 'sequences' in result and result['sequences']:
                    generated_seq = result['sequences'][0]
                    
                    # Ensure motif is preserved
                    motif_preserved = motif_data.motif_sequence in generated_seq
                    
                    return {
                        'full_sequence': generated_seq,
                        'motif_preserved': motif_preserved,
                        'scaffold_length': len(generated_seq) - len(motif_data.motif_sequence),
                        'method': 'flowflow_server',
                        'server_response': result
                    }
                else:
                    print(f"   ⚠️ {self.name} server returned no sequences")
                    return self._fallback_generation(motif_data, scaffold_length)
            else:
                print(f"   ⚠️ {self.name} server error: {response.status_code}")
                return self._fallback_generation(motif_data, scaffold_length)
                
        except Exception as e:
            print(f"   ⚠️ {self.name} generation failed: {e}")
            return self._fallback_generation(motif_data, scaffold_length)
    
    def _fallback_generation(self, motif_data, scaffold_length: int) -> Dict:
        """Fallback generation when server is not available."""
        import random
        
        # Generate flow-biased scaffold (more structured sequences)
        structured_aa = "ADEFHIKLNQRSTVWY"  # Helix/sheet forming
        
        # Calculate actual scaffold length
        target_length = len(motif_data.full_sequence)
        actual_scaffold_length = target_length - len(motif_data.motif_sequence)
        
        # Insert motif at random position
        motif_pos = random.randint(0, max(0, actual_scaffold_length - len(motif_data.motif_sequence)))
        
        # Generate flow-biased scaffold
        left_scaffold = ''.join(random.choices(structured_aa, k=motif_pos))
        right_scaffold = ''.join(random.choices(structured_aa, k=actual_scaffold_length - motif_pos))
        
        full_sequence = left_scaffold + motif_data.motif_sequence + right_scaffold
        
        return {
            'full_sequence': full_sequence,
            'motif_preserved': motif_data.motif_sequence in full_sequence,
            'scaffold_length': actual_scaffold_length,
            'method': 'flowflow_fallback'
        }
    
    def get_name(self) -> str:
        return self.name

