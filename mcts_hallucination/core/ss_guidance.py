"""
Secondary Structure Guidance Module for Hallucination MCTS.

This module provides DSSP-based secondary structure analysis and
four SS-guided sequence update policies for protein hallucination.

Variants:
1. beta_lock_reinit_helix: Random init + DSSP beta-lock, helix reinit
2. x_init_helix_pg: X init + helix-targeted P/G injection after MPNN
3. x_init_beta_template: X init + first-iter beta template conditioning
4. x_init_ligand_first: X init + first-iter ligand conditioning (ATP)
"""

import json
import random
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# DSSP binary path on /net/scratch
DSSP_BINARY_PATH = "/net/scratch/caom/dssp_env/bin/mkdssp"


@dataclass
class SSGuidanceConfig:
    """Configuration for secondary structure guidance."""
    ss_guidance: str = "none"
    dssp_backend: str = "mkdssp"
    ss_require_dssp: bool = True
    beta_min_len: int = 3
    helix_min_len: int = 4
    pg_inject_prob: float = 0.1
    pg_inject_targets: str = "PG"
    helix_reinit: str = "mask_x"
    beta_template_path: Optional[str] = None
    template_persist: bool = False
    ligand_path: Optional[str] = None
    first_iter_only: bool = True
    results_dir: str = "/net/scratch/caom/ss_guidance_results"


@dataclass
class DSSPResult:
    """Result from DSSP analysis."""
    per_residue_ss: List[str]
    helix_positions: Set[int]
    beta_positions: Set[int]
    helix_segments: List[Tuple[int, int]]
    beta_segments: List[Tuple[int, int]]
    coil_positions: Set[int]


@dataclass
class EditLogEntry:
    """Log entry for sequence edits."""
    position: int
    original: str
    new: str
    reason: str
    variant: str


class SSGuidance:
    """Secondary structure guidance for hallucination MCTS."""
    
    def __init__(self, config: SSGuidanceConfig):
        self.config = config
        self.dssp_available = self._check_dssp_available()
        
        if not self.dssp_available:
            if config.ss_require_dssp:
                raise RuntimeError(
                    f"DSSP is required but not available. "
                    f"Expected mkdssp at: {DSSP_BINARY_PATH}\n"
                    f"Install: micromamba create -p /net/scratch/caom/dssp_env -c conda-forge dssp"
                )
            else:
                print(f"Warning: DSSP not available - SS guidance will be skipped")
    
    def _check_dssp_available(self) -> bool:
        """Check if DSSP is available at the expected path."""
        dssp_path = Path(DSSP_BINARY_PATH)
        if not dssp_path.exists():
            return False
        try:
            result = subprocess.run(
                [str(dssp_path), "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, PermissionError, OSError) as e:
            print(f"Warning: DSSP check failed: {e}")
            return False
    
    def run_dssp(self, pdb_path: str, sequence_length: int) -> Optional[DSSPResult]:
        """Run DSSP on a PDB file and extract secondary structure."""
        if not self.dssp_available:
            return None
        try:
            result = subprocess.run(
                [DSSP_BINARY_PATH, pdb_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode != 0:
                print(f"Warning: DSSP failed: {result.stderr[:200]}")
                return None
            return self._parse_dssp_output(result.stdout, sequence_length)
        except subprocess.TimeoutExpired:
            print("Warning: DSSP timed out")
            return None
        except Exception as e:
            print(f"Warning: DSSP error: {e}")
            return None
    
    def _parse_dssp_output(self, dssp_output: str, sequence_length: int) -> DSSPResult:
        """Parse DSSP output to extract secondary structure."""
        per_residue_ss = ['-'] * sequence_length
        helix_positions = set()
        beta_positions = set()
        coil_positions = set()
        in_data = False
        residue_idx = 0
        
        for line in dssp_output.split('\n'):
            if line.startswith('  #'):
                in_data = True
                continue
            if not in_data or len(line) < 17:
                continue
            if len(line) > 13 and line[13] == '!':
                continue
            ss_char = line[16] if len(line) > 16 else '-'
            if residue_idx < sequence_length:
                per_residue_ss[residue_idx] = ss_char
                if ss_char in ('H', 'G', 'I'):
                    helix_positions.add(residue_idx)
                elif ss_char in ('E', 'B'):
                    beta_positions.add(residue_idx)
                else:
                    coil_positions.add(residue_idx)
                residue_idx += 1
        
        for i in range(residue_idx, sequence_length):
            coil_positions.add(i)
        
        helix_segments = self._extract_segments(helix_positions, self.config.helix_min_len)
        beta_segments = self._extract_segments(beta_positions, self.config.beta_min_len)
        
        return DSSPResult(
            per_residue_ss=per_residue_ss,
            helix_positions=helix_positions,
            beta_positions=beta_positions,
            helix_segments=helix_segments,
            beta_segments=beta_segments,
            coil_positions=coil_positions,
        )
    
    def _extract_segments(self, positions: Set[int], min_len: int) -> List[Tuple[int, int]]:
        """Extract contiguous segments from a set of positions."""
        if not positions:
            return []
        sorted_pos = sorted(positions)
        segments = []
        start = sorted_pos[0]
        end = sorted_pos[0]
        for pos in sorted_pos[1:]:
            if pos == end + 1:
                end = pos
            else:
                if end - start + 1 >= min_len:
                    segments.append((start, end))
                start = pos
                end = pos
        if end - start + 1 >= min_len:
            segments.append((start, end))
        return segments
    
    def apply_variant1_beta_lock_reinit_helix(
        self, sequence: str, dssp_result: DSSPResult
    ) -> Tuple[str, List[EditLogEntry]]:
        """Variant 1: Beta-lock + helix reinit."""
        edit_log = []
        seq_list = list(sequence)
        helix_in_segments = set()
        for start, end in dssp_result.helix_segments:
            helix_in_segments.update(range(start, end + 1))
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        for pos in helix_in_segments:
            if pos < len(seq_list):
                original = seq_list[pos]
                if self.config.helix_reinit == "mask_x":
                    new_char = 'X'
                else:
                    new_char = random.choice(amino_acids)
                if original != new_char:
                    seq_list[pos] = new_char
                    edit_log.append(EditLogEntry(
                        position=pos, original=original, new=new_char,
                        reason="helix_reinit", variant="beta_lock_reinit_helix"
                    ))
        return ''.join(seq_list), edit_log
    
    def apply_variant2_helix_pg_injection(
        self, sequence: str, dssp_result: DSSPResult
    ) -> Tuple[str, List[EditLogEntry]]:
        """Variant 2: P/G injection in helix regions after MPNN."""
        edit_log = []
        seq_list = list(sequence)
        helix_in_segments = set()
        for start, end in dssp_result.helix_segments:
            helix_in_segments.update(range(start, end + 1))
        targets = list(self.config.pg_inject_targets)
        for pos in helix_in_segments:
            if pos < len(seq_list) and random.random() < self.config.pg_inject_prob:
                original = seq_list[pos]
                new_char = random.choice(targets)
                if original != new_char:
                    seq_list[pos] = new_char
                    edit_log.append(EditLogEntry(
                        position=pos, original=original, new=new_char,
                        reason="pg_injection", variant="x_init_helix_pg"
                    ))
        return ''.join(seq_list), edit_log
    
    def get_template_conditioning(self, iteration: int) -> Optional[Dict]:
        """Get template conditioning for Variant 3."""
        if self.config.ss_guidance != "x_init_beta_template":
            return None
        if not self.config.beta_template_path:
            return None
        if iteration > 0 and not self.config.template_persist:
            return None
        return {"template_path": self.config.beta_template_path, "iteration": iteration}
    
    def get_ligand_conditioning(self, iteration: int) -> Optional[Dict]:
        """Get ligand conditioning for Variant 4."""
        if self.config.ss_guidance != "x_init_ligand_first":
            return None
        if not self.config.ligand_path:
            return None
        if iteration > 0 and self.config.first_iter_only:
            return None
        return {"ligand_path": self.config.ligand_path, "iteration": iteration}
    
    def save_dssp_json(self, dssp_result: DSSPResult, output_path: str):
        """Save DSSP result to JSON file."""
        data = {
            "per_residue_ss": dssp_result.per_residue_ss,
            "helix_positions": sorted(dssp_result.helix_positions),
            "beta_positions": sorted(dssp_result.beta_positions),
            "helix_segments": dssp_result.helix_segments,
            "beta_segments": dssp_result.beta_segments,
            "coil_positions": sorted(dssp_result.coil_positions),
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_edit_log(self, edit_log: List[EditLogEntry], output_path: str):
        """Save edit log to JSON file."""
        data = [asdict(entry) for entry in edit_log]
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


def create_run_directory(config: SSGuidanceConfig, seed: Optional[int] = None) -> Path:
    """Create a run directory for SS guidance artifacts."""
    import datetime
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        git_hash = result.stdout.strip() if result.returncode == 0 else "unknown"
    except:
        git_hash = "unknown"
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{git_hash}"
    if seed is not None:
        run_id += f"_s{seed}"
    
    run_dir = Path(config.results_dir) / config.ss_guidance / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    config_data = asdict(config)
    config_data["run_id"] = run_id
    config_data["git_hash"] = git_hash
    config_data["seed"] = seed
    
    with open(run_dir / "config.json", 'w') as f:
        json.dump(config_data, f, indent=2)
    
    return run_dir
