"""
ESMFold Integration for structure hallucination.

Provides a lightweight wrapper around the HuggingFace ESMFold model so the
hallucination expert can run without AlphaFold3/ABCFold.
"""

from typing import Dict, Optional
import numpy as np

try:
    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding
except ImportError:
    torch = None
    AutoTokenizer = None
    EsmForProteinFolding = None


class ESMFoldIntegration:
    """Predict structures using the pretrained ESMFold model."""
    
    def __init__(
        self,
        model_name: str = "facebook/esmfold_v1",
        device: str = "cuda",
        use_mock: bool = False,
        max_length: int = 1024,
        allow_fallback: bool = False,
    ):
        """
        Args:
            model_name: HuggingFace model identifier.
            device: Device to run inference on ("cuda" or "cpu").
            use_mock: If True, skip model loading and return synthetic structures.
            max_length: Maximum supported sequence length.
        """
        self.model_name = model_name
        self.requested_device = device
        self.device = None
        backend_missing = torch is None or AutoTokenizer is None
        if backend_missing and not use_mock:
            raise ImportError("Torch/transformers are required for real ESMFold inference. Install them or set use_mock=True.")
        self.use_mock = use_mock or backend_missing
        self.max_length = max_length
        self.allow_fallback = allow_fallback
        self.model: Optional[EsmForProteinFolding] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        
        if not self.use_mock:
            self._load_model()
        else:
            if torch is None:
                print("⚠️ Torch/transformers unavailable. ESMFoldIntegration running in mock mode.")
            else:
                print("🔧 ESMFold Integration initialized (MOCK MODE)")
    
    def _load_model(self):
        """Load the ESMFold model from HuggingFace."""
        try:
            self.device = torch.device(
                self.requested_device if self.requested_device == "cpu" or torch.cuda.is_available()
                else "cpu"
            )
            self.model = EsmForProteinFolding.from_pretrained(self.model_name)
            self.model = self.model.to(self.device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print(f"🔧 ESMFold Integration initialized (REAL MODE on {self.device})")
        except Exception as e:
            if self.allow_fallback or self.use_mock:
                print(f"⚠️ Failed to load ESMFold ({e}). Falling back to mock mode.")
                self.model = None
                self.tokenizer = None
                self.use_mock = True
            else:
                raise RuntimeError(f"Failed to load ESMFold model '{self.model_name}': {e}") from e
    
    def predict_structure(self, sequence: str) -> Dict:
        """Predict structure for a sequence using ESMFold."""
        clean_sequence = sequence.replace("X", "A")
        
        if self.use_mock or self.model is None or self.tokenizer is None:
            return self._mock_predict(clean_sequence)
        
        if len(clean_sequence) > self.max_length:
            raise ValueError(
                f"Sequence length {len(clean_sequence)} exceeds ESMFold limit ({self.max_length})."
            )
        
        try:
            tokenized = self.tokenizer(
                clean_sequence,
                return_tensors="pt",
                add_special_tokens=False,
            )
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            
            with torch.no_grad():
                output = self.model(tokenized["input_ids"])
            
            coords = self._extract_ca_coordinates(output["positions"])
            confidence = self._extract_plddt(output)
            
            return {
                "coordinates": coords,
                "confidence": confidence,
                "pae_mean": float(np.nan),
            }
        except Exception as e:
            if self.allow_fallback or self.use_mock:
                print(f"⚠️ ESMFold prediction failed: {e}. Falling back to mock output.")
                return self._mock_predict(clean_sequence)
            raise RuntimeError(f"ESMFold prediction failed: {e}") from e
    
    def _extract_ca_coordinates(self, positions_tensor) -> np.ndarray:
        """Convert raw ESMFold outputs into CA coordinate arrays."""
        positions = positions_tensor.detach().cpu()
        
        if positions.dim() == 5:
            # (batch, extra_dim, seq_len, atoms, 3)
            ca_coords = positions[0, 0, :, 1, :].numpy()
        elif positions.dim() == 4:
            # (batch, seq_len, atoms, 3)
            ca_coords = positions[0, :, 1, :].numpy()
        elif positions.dim() == 3:
            # (seq_len, atoms, 3)
            ca_coords = positions[:, 1, :].numpy()
        else:
            arr = positions.numpy()
            if arr.ndim > 2:
                ca_coords = arr.reshape(-1, arr.shape[-1])
            else:
                ca_coords = arr
        
        return ca_coords
    
    def _extract_plddt(self, output) -> np.ndarray:
        """Extract per-residue confidences from ESMFold output."""
        if "plddt" not in output:
            # Provide a constant confidence if unavailable
            return np.full((output["positions"].shape[-3],), 70.0)
        
        plddt = output["plddt"].detach().cpu().numpy()
        if plddt.ndim > 1:
            plddt = plddt.flatten()
        
        if plddt.max() <= 1.5:
            plddt = plddt * 100.0
        
        return plddt
    
    def _mock_predict(self, sequence: str) -> Dict:
        """Return synthetic coordinates/confidence for testing."""
        n = len(sequence)
        coords = np.cumsum(np.random.randn(n, 3) * 1.5, axis=0)
        coords += np.random.randn(n, 3) * 2.0
        confidence = 70 + 20 * np.exp(-((np.arange(n) - n / 2) ** 2) / (n / 4) ** 2)
        confidence += np.random.randn(n) * 5
        confidence = np.clip(confidence, 40, 95)
        
        return {
            "coordinates": coords,
            "confidence": confidence,
            "pae_mean": float(np.nan),
        }
