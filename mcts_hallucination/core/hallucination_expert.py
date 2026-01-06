"""
Hallucination Expert for MCTS

This expert integrates AF3 (via ABCFold) + ProteinMPNN as a two-step process:
1. AF3 hallucinates structure from masked sequence
2. ProteinMPNN designs sequence from hallucinated structure

This plugs into the existing GeneralMCTS as a new expert type.

Supported backends:
- Structure: ABCFold (AF3/Boltz/Chai-1), ESMFold
- Inverse Folding: ProteinMPNN, NA-MPNN (nucleic acids)
"""

import numpy as np
from typing import Dict, Set, Optional, List
import sys
import os
from pathlib import Path

# Import the integrations
from core.abcfold_integration import ABCFoldIntegration
from core.esmfold_integration import ESMFoldIntegration

# Optional: Import NA-MPNN integration
try:
    from core.nampnn_integration import NAMPNNIntegration
    NAMPNN_AVAILABLE = True
except ImportError:
    NAMPNNIntegration = None
    NAMPNN_AVAILABLE = False

# Ensure the parent package (mcts_diffusion_finetune) is on sys.path so we can
# reuse the real ProteinMPNN implementation when available.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Add the mcts_diffusion_finetune/core directory directly to avoid __init__.py import chain
CORE_DIR = REPO_ROOT / "mcts_diffusion_finetune" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

_REAL_PROTEINMPNN_ERROR = None
try:
    # Import directly from the file to avoid circular imports via __init__.py
    from proteinmpnn_real import RealProteinMPNNExpert
    REAL_PROTEINMPNN_AVAILABLE = True
except Exception as exc:  # pragma: no cover - best-effort import
    RealProteinMPNNExpert = None
    REAL_PROTEINMPNN_AVAILABLE = False
    _REAL_PROTEINMPNN_ERROR = exc


class ProteinMPNNIntegration:
    """ProteinMPNN wrapper for the hallucination pipeline."""
    
    def __init__(
        self,
        use_real: bool = True,
        device: str = "cuda",
        temperature: float = 1.0,
    ):
        self.device = device
        self.temperature = temperature
        if use_real and not REAL_PROTEINMPNN_AVAILABLE:
            raise ImportError(
                "Real ProteinMPNN is not available. "
                "Set the PROTEINMPNN_PATH environment variable to the directory "
                "containing third_party/proteinpmnn and ensure dependencies are installed."
            ) from _REAL_PROTEINMPNN_ERROR
        self.use_real = use_real
        self.real_expert: Optional[RealProteinMPNNExpert] = None
        
        if self.use_real:
            try:
                self.real_expert = RealProteinMPNNExpert(
                    device=device,
                    temperature=temperature,
                )
                self.real_expert.load_model()
                print("🔧 ProteinMPNN Integration initialized (REAL MODE)")
            except Exception as exc:
                raise RuntimeError(f"Failed to initialize real ProteinMPNN: {exc}") from exc
        
        if not self.use_real:
            print("🔧 ProteinMPNN Integration initialized (MOCK MODE)")
    
    def design_sequence(self, coordinates: np.ndarray, masked_sequence: Optional[str] = None) -> str:
        """Design sequence from hallucinated coordinates."""
        if self.use_real and self.real_expert is not None:
            if masked_sequence is None:
                raise ValueError("masked_sequence is required for real ProteinMPNN generation.")
            sequences = self.real_expert.generate_sequences_from_coords(
                masked_sequence=masked_sequence,
                coords=coordinates,
                num_samples=1,
            )
            if not sequences:
                raise RuntimeError("Real ProteinMPNN failed to return a sequence.")
            return sequences[0]
        
        # Mock fallback: generate random sequence of same length
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        n = len(coordinates)
        return ''.join(np.random.choice(list(amino_acids), size=n))


class HallucinationExpert:
    """
    Expert that uses AF3 hallucination + ProteinMPNN inverse folding.
    
    This can be added to GeneralMCTS.external_experts list.
    
    Supported structure backends:
    - "abcfold": ABCFold wrapper (AF3, Boltz, Chai-1)
    - "esmfold": ESMFold from HuggingFace
    
    Supported inverse folding backends:
    - "proteinmpnn": Standard ProteinMPNN (proteins only)
    - "nampnn": NA-MPNN (supports DNA/RNA)
    
    Fallback behavior:
    - If fallback_to_protein_mpnn=True and input is protein-only,
      NA-MPNN will automatically fallback to ProteinMPNN for better performance.
    """
    
    def __init__(
        self,
        model_params: str = None,
        use_mock: bool = False,
        structure_backend: str = "abcfold",
        inverse_folding_backend: str = "proteinmpnn",
        molecule_type: str = "protein",
        fallback_to_protein_mpnn: bool = True,
        abcfold_engine: str = "af3",
        abcfold_database_dir: str = None,
        abcfold_use_mmseqs: bool = True,
        esmfold_model_name: str = "facebook/esmfold_v1",
        esmfold_device: str = "cuda",
        esmfold_max_length: int = 1024,
        use_real_proteinmpnn: bool = True,
        proteinmpnn_device: str = "cuda",
        proteinmpnn_temperature: float = 1.0,
        nampnn_path: str = None,
        nampnn_temperature: float = 0.1,
    ):
        """
        Initialize hallucination expert.
        
        Args:
            model_params: Path to AF3 model parameters (for real mode)
            use_mock: Use mock mode for testing (default: False)
            structure_backend: "abcfold" or "esmfold"
            inverse_folding_backend: "proteinmpnn" or "nampnn"
            molecule_type: "protein", "dna", "rna", or "complex"
            fallback_to_protein_mpnn: If True and using NA-MPNN, automatically
                fallback to ProteinMPNN when input contains only protein chains
            abcfold_engine: Which ABCFold engine to run ("af3", "boltz", "chai1")
            abcfold_database_dir: Optional AF3 database directory
            abcfold_use_mmseqs: Whether to enable MMseqs2 flag for ABCFold
            esmfold_model_name: HuggingFace identifier for ESMFold model
            esmfold_device: Device for ESMFold inference ("cuda" or "cpu")
            esmfold_max_length: Maximum ESMFold sequence length
            use_real_proteinmpnn: Use the real ProteinMPNN inverse folding model
            proteinmpnn_device: Device for ProteinMPNN inference
            proteinmpnn_temperature: Sampling temperature for ProteinMPNN
            nampnn_path: Path to NA-MPNN repository
            nampnn_temperature: Sampling temperature for NA-MPNN
        """
        self.structure_backend = structure_backend.lower()
        self.inverse_folding_backend = inverse_folding_backend.lower()
        self.molecule_type = molecule_type.lower()
        self.use_mock = use_mock
        self.fallback_to_protein_mpnn = fallback_to_protein_mpnn
        
        # Initialize structure predictor
        if self.structure_backend == "abcfold":
            self.structure_predictor = ABCFoldIntegration(
                model_params=model_params,
                database_dir=abcfold_database_dir,
                use_mmseqs=abcfold_use_mmseqs,
                use_mock=use_mock,
                engine=abcfold_engine,
                molecule_type=molecule_type,  # Pass molecule type for correct FASTA/JSON format
            )
            self.backend_label = f"ABCFold ({abcfold_engine.upper()})"
        elif self.structure_backend == "esmfold":
            self.structure_predictor = ESMFoldIntegration(
                model_name=esmfold_model_name,
                device=esmfold_device,
                use_mock=use_mock,
                max_length=esmfold_max_length,
            )
            self.backend_label = "ESMFold"
        else:
            raise ValueError(
                f"Unsupported structure backend '{structure_backend}'. "
                f"Choose from: abcfold, esmfold"
            )
        
        # Initialize inverse folding backend
        self._proteinmpnn_fallback = None  # For protein-only fallback
        
        if self.inverse_folding_backend == "proteinmpnn":
            self.inverse_folder = ProteinMPNNIntegration(
                use_real=use_real_proteinmpnn,
                device=proteinmpnn_device,
                temperature=proteinmpnn_temperature,
            )
            self.inverse_folder_label = "ProteinMPNN"
        elif self.inverse_folding_backend == "nampnn":
            if not NAMPNN_AVAILABLE:
                raise ImportError(
                    "NA-MPNN integration not available. "
                    "Check that nampnn_integration.py exists and has no import errors."
                )
            self.inverse_folder = NAMPNNIntegration(
                nampnn_path=nampnn_path,
                use_mock=use_mock,
                temperature=nampnn_temperature,
            )
            self.inverse_folder_label = "NA-MPNN"
            
            # Initialize ProteinMPNN fallback for protein-only cases
            if self.fallback_to_protein_mpnn:
                self._proteinmpnn_fallback = ProteinMPNNIntegration(
                    use_real=use_real_proteinmpnn,
                    device=proteinmpnn_device,
                    temperature=proteinmpnn_temperature,
                )
        else:
            raise ValueError(
                f"Unsupported inverse folding backend '{inverse_folding_backend}'. "
                f"Choose from: proteinmpnn, nampnn"
            )
        
        # For backwards compatibility, also expose as self.proteinmpnn
        self.proteinmpnn = self.inverse_folder
        
        self.name = "hallucination"
        
        print(f"🔧 HallucinationExpert initialized")
        print(f"   Structure: {self.backend_label}")
        print(f"   Inverse Folding: {self.inverse_folder_label}")
        print(f"   Molecule Type: {self.molecule_type}")
        if self._proteinmpnn_fallback:
            print(f"   Protein-only Fallback: ProteinMPNN")
    
    def get_name(self) -> str:
        """Return expert name."""
        return self.name
    
    def generate_candidate(
        self, 
        sequence: str, 
        masked_positions: Set[int],
        coordinates: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict:
        """
        Generate candidate using AF3 + ProteinMPNN pipeline.
        
        Process:
        1. Create masked sequence (X at masked positions)
        2. AF3 hallucinates structure from masked sequence
        3. ProteinMPNN designs sequence from hallucinated structure
        4. Return candidate with sequence, coordinates, and confidence
        
        Args:
            sequence: Current sequence
            masked_positions: Positions to re-design
            coordinates: Optional current coordinates (not used for hallucination)
            
        Returns:
            Dictionary with:
                - sequence: New designed sequence
                - coordinates: Hallucinated structure coordinates
                - confidence_scores: AF3 confidence scores (pLDDT-like)
                - expert: Expert name
                - entropy: Model uncertainty
        """
        try:
            # Step 1: Create masked sequence
            masked_seq = self._create_masked_sequence(sequence, masked_positions)
            print(f"      🎭 Hallucination: {len(masked_positions)} positions masked")
            
            # Step 2: Structure hallucination
            print(f"      🔮 {self.backend_label}: Predicting structure...")
            structure_result = self.structure_predictor.predict_structure(masked_seq)
            hallucinated_coords = structure_result['coordinates']
            confidence_scores = structure_result['confidence']
            mean_plddt = np.mean(confidence_scores)
            print(f"      ✅ {self.backend_label}: Structure predicted (mean pLDDT: {mean_plddt:.1f})")
            
            # Step 3: Inverse folding (ProteinMPNN or NA-MPNN)
            print(f"      🧬 {self.inverse_folder_label}: Designing sequence...")
            designed_sequence = self.inverse_folder.design_sequence(
                hallucinated_coords,
                masked_sequence=masked_seq,
            )
            print(f"      ✅ {self.inverse_folder_label}: Sequence designed")
            
            # Step 4: Compute entropy from confidence variance
            entropy = self._compute_entropy(confidence_scores)
            
            pae_value = structure_result.get('pae_mean', 0.0)
            if isinstance(pae_value, (float, int)) and not np.isnan(pae_value):
                pae_mean = float(pae_value)
            else:
                pae_mean = 0.0
            
            return {
                'sequence': designed_sequence,
                'coordinates': hallucinated_coords,
                'confidence_scores': confidence_scores.tolist() if isinstance(confidence_scores, np.ndarray) else confidence_scores,
                'plddt_scores': confidence_scores.tolist() if isinstance(confidence_scores, np.ndarray) else confidence_scores,
                'expert': self.name,
                'entropy': entropy,
                'mean_plddt': mean_plddt,
                'pae_mean': pae_mean
            }
            
        except Exception as e:
            print(f"      ❌ Hallucination expert failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_masked_sequence(self, sequence: str, masked_positions: Set[int]) -> str:
        """Create masked sequence with X at masked positions."""
        masked = list(sequence)
        for pos in masked_positions:
            if pos < len(masked):
                masked[pos] = 'X'
        return ''.join(masked)
    
    def _compute_entropy(self, confidence_scores: np.ndarray) -> float:
        """
        Compute entropy from confidence score variance.
        
        Higher variance = higher uncertainty = higher entropy
        """
        if confidence_scores is None or len(confidence_scores) == 0:
            return 0.5
        
        # Convert to numpy if needed
        if not isinstance(confidence_scores, np.ndarray):
            confidence_scores = np.array(confidence_scores)
        
        # Compute variance and normalize to [0, 1]
        variance = np.var(confidence_scores)
        # Typical pLDDT variance is 0-400, normalize
        normalized_entropy = min(1.0, variance / 400.0)
        
        return normalized_entropy
    
    def compute_entropy(
        self, 
        sequence: str, 
        masked_positions: Set[int],
        **kwargs
    ) -> float:
        """
        Compute predictive entropy for PH-UCT.
        
        For hallucination, we can't easily compute entropy without running AF3,
        so we return a default value. The actual entropy is computed during generation.
        """
        # Default entropy for hallucination (medium uncertainty)
        return 0.5
    
    def is_protein_only(self, complex_input: "ComplexInput") -> bool:
        """
        Check if a ComplexInput contains only protein chains.
        
        Used to determine whether to fallback to ProteinMPNN.
        """
        from core.complex_input import ProteinChain
        
        for chain in complex_input.chains.values():
            if not isinstance(chain, ProteinChain):
                return False
        return True
    
    def get_inverse_folder(self, complex_input: "ComplexInput" = None):
        """
        Get the appropriate inverse folder based on input composition.
        
        If fallback_to_protein_mpnn=True and input is protein-only,
        returns ProteinMPNN. Otherwise returns the configured inverse folder.
        """
        if self._proteinmpnn_fallback and complex_input:
            if self.is_protein_only(complex_input):
                print("      📌 Protein-only input detected, using ProteinMPNN fallback")
                return self._proteinmpnn_fallback
        return self.inverse_folder


def create_hallucination_expert(
    model_params: str = None,
    use_mock: bool = False,
    structure_backend: str = "abcfold",
    inverse_folding_backend: str = "proteinmpnn",
    molecule_type: str = "protein",
    fallback_to_protein_mpnn: bool = True,
    **kwargs,
) -> HallucinationExpert:
    """
    Factory function to create hallucination expert.
    
    Args:
        model_params: Path to AF3 model parameters (for real mode)
        use_mock: Use mock mode for testing (default: False)
        structure_backend: "abcfold" or "esmfold"
        inverse_folding_backend: "proteinmpnn" or "nampnn"
        molecule_type: "protein", "dna", "rna", or "complex"
        fallback_to_protein_mpnn: When using NA-MPNN, fallback to 
            ProteinMPNN for protein-only inputs (default: True)
    
    Usage:
        # Basic protein design (mock mode for testing)
        expert = create_hallucination_expert(use_mock=True)
        
        # With ABCFold + NA-MPNN for DNA/RNA design (auto-fallback for protein-only)
        expert = create_hallucination_expert(
            structure_backend="abcfold",
            inverse_folding_backend="nampnn",
            molecule_type="complex",
            fallback_to_protein_mpnn=True,
            use_mock=True,
        )
        
        # Real mode with AF3
        expert = create_hallucination_expert(
            model_params="/path/to/af3/params",
            use_mock=False
        )
        
        # Add to MCTS
        mcts = GeneralMCTS(
            dplm2_integration=dplm2,
            external_experts=[expert],
            ablation_mode="single_expert",
            single_expert_id=3
        )
    """
    return HallucinationExpert(
        model_params=model_params,
        use_mock=use_mock,
        structure_backend=structure_backend,
        inverse_folding_backend=inverse_folding_backend,
        molecule_type=molecule_type,
        fallback_to_protein_mpnn=fallback_to_protein_mpnn,
        **kwargs,
    )
