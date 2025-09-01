# Minimal, robust DPLM-2 integration aligned with generate_dplm2_patched_v2.py

import os, sys
import torch
import numpy as np
from typing import Dict, List, Optional

# ---- optional envs ----
os.environ.setdefault('HF_HOME', '/net/scratch/caom/.cache/huggingface')
os.environ.setdefault('TRANSFORMERS_CACHE', '/net/scratch/caom/.cache/huggingface/transformers')
os.environ.setdefault('TORCH_HOME', '/net/scratch/caom/.cache/torch')

# ---- Mock CUDA extension to avoid import errors ----
def patch_cuda_extension():
    """Mock missing CUDA extension to allow DPLM-2 imports"""
    import sys
    from types import ModuleType
    
    # Create mock CUDA module
    mock_cuda = ModuleType('attn_core_inplace_cuda')
    mock_cuda.attention_core = lambda *args, **kwargs: None
    sys.modules['attn_core_inplace_cuda'] = mock_cuda
    print("✓ CUDA extension mocked")

patch_cuda_extension()

# ---- import DPLM-2 ----
try:
    from byprot.models.dplm2 import MultimodalDiffusionProteinLanguageModel as DPLM2
except Exception:
    from byprot.models.dplm2 import DPLM2Bit as DPLM2  # fallback

class DPLM2IntegrationClean:
    """
    Minimal & robust DPLM-2 integration:
    - ALWAYS go string->tokenizer->batch_encode_plus
    - NEVER rely on truthiness of numpy/list
    - Baseline: masked_sequence=None => 'X'*L
    - struct_seq: normalize to list[str], then "<cls_struct>" + "".join(list) + "<eos_struct>"
    """

    def __init__(self, model_name: str = "airkingbd/dplm2_150m", device: Optional[str] = None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = DPLM2.from_pretrained(model_name, from_huggingface=True).to(self.device).eval()
        self.tok = self.model.tokenizer

    # ---------- helpers ----------
    @staticmethod
    def _is_seq_container(x):
        return isinstance(x, (list, tuple, np.ndarray))

    def _normalize_struct_seq_to_list_str(self, struct_seq) -> List[str]:
        """struct_seq can be:
        - '159,160,161,...' (str with commas)
        - ['159','160',...]/[159,160,...]/np.array([...])
        Returns ['159','160',...]
        """
        if struct_seq is None:
            raise ValueError("struct_seq is None")

        # str with commas: split
        if isinstance(struct_seq, str) and ("," in struct_seq):
            parts = [p.strip() for p in struct_seq.split(",") if p.strip()]
            return parts

        # container type: convert each to string
        if self._is_seq_container(struct_seq):
            return [str(int(x)) for x in list(struct_seq)]

        # str without commas (rare): error
        if isinstance(struct_seq, str):
            raise ValueError(
                "struct_seq is a plain string without commas. Expected '159,160,...' or a list/array."
            )

        # other types
        raise ValueError(f"Unsupported struct_seq type: {type(struct_seq)}")

    def _build_struct_text(self, struct_seq) -> str:
        ids = self._normalize_struct_seq_to_list_str(struct_seq)
        # <cls_struct> + no commas + <eos_struct>
        return self.tok.struct_cls_token + "".join(ids) + self.tok.struct_eos_token

    def _build_aa_text(self, masked_sequence: Optional[str], L: int) -> str:
        """masked_sequence=None -> baseline (full masking)"""
        if masked_sequence is None:
            body = self.tok.aa_mask_token * L
        else:
            s = str(masked_sequence)
            body = "".join(self.tok.aa_mask_token if c == "X" else c for c in s)
        return self.tok.aa_cls_token + body + self.tok.aa_eos_token

    def _string_encode(self, texts: List[str]):
        # Consistent with generate script: longest padding + return_tensors='pt'
        return self.tok.batch_encode_plus(
            texts,
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )

    # ---------- public APIs ----------
    def fill_masked_positions(
        self,
        structure: Dict,
        masked_sequence: Optional[str] = None,
        target_length: Optional[int] = None,
        max_iter: int = 100,
        temperature: float = 1.0,
        unmasking_strategy: str = "deterministic",
        sampling_strategy: str = "argmax",
    ) -> str:
        """
        structure must have struct_seq or struct_ids.
        - struct_seq: '159,160,...' or list/ndarray
        - masked_sequence=None => baseline: full masking (length matches structure tokens)
        """
        # 1) Get struct_seq
        struct_seq = None
        if "struct_seq" in structure and structure["struct_seq"] is not None:
            struct_seq = structure["struct_seq"]
        elif "struct_ids" in structure and structure["struct_ids"] is not None:
            struct_seq = structure["struct_ids"]
        else:
            raise ValueError("Need structure['struct_seq'] or structure['struct_ids'].")

        # 2) Build structure text (strict no commas) + AA text
        struct_text = self._build_struct_text(struct_seq)
        L = len(self._normalize_struct_seq_to_list_str(struct_seq))
        aa_text = self._build_aa_text(masked_sequence, L)

        # 3) encode -> concatenate
        batch_struct = self._string_encode([struct_text])
        batch_aa = self._string_encode([aa_text])
        input_tokens = torch.cat([batch_struct["input_ids"], batch_aa["input_ids"]], dim=1).to(self.device)

        # 4) Modality masks (consistent with generate script)
        type_ids = self.model.get_modality_type(input_tokens)  # 0=struct, 1=aa
        non_special = self.model.get_non_special_symbol_mask(input_tokens)
        struct_type, aa_type = 0, 1

        # Freeze: all structure tokens + all special tokens
        partial_mask = (type_ids == struct_type) | (~non_special)

        # baseline: set AA body positions to mask id first (like script)
        if masked_sequence is None:
            mask_id = self.tok._token_to_id[self.tok.aa_mask_token]
            input_tokens.masked_fill_((type_ids == aa_type) & non_special, mask_id)
        else:
            # partial filling: freeze non-X positions too
            ms = str(masked_sequence)
            # struct segment length
            Ls = batch_struct["input_ids"].shape[1]
            # AA segment: <cls_aa> BODY <eos_aa>
            for i, c in enumerate(ms):
                if c != "X":
                    partial_mask[0, Ls + 1 + i] = True

        # 5) Generate
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = self.model.generate(
                input_tokens=input_tokens,
                partial_masks=partial_mask,
                max_iter=max_iter,
                temperature=temperature,
                unmasking_strategy=unmasking_strategy,
                sampling_strategy=sampling_strategy,
            )

        # 6) Decode (strict from modality=AA only)
        tokens = out["output_tokens"][0]  # [T]
        out_type = self.model.get_modality_type(tokens.unsqueeze(0))[0]  # [T]
        aa_idx = (out_type == 1).nonzero(as_tuple=False).flatten()
        if aa_idx.numel() == 0:
            # fallback: try to extract from all tokens
            decoded_all = self.tok.decode(tokens.cpu().tolist())
            seq = "".join([c for c in decoded_all if c in "ACDEFGHIKLMNPQRSTVWY"])
            return seq

        aa_tokens = tokens[aa_idx].cpu().tolist()
        seq = self.tok.decode(aa_tokens)
        # remove special tokens
        seq = seq.replace(self.tok.aa_cls_token, "").replace(self.tok.aa_eos_token, "")
        # only keep 20 amino acids
        seq = "".join([c for c in seq if c in "ACDEFGHIKLMNPQRSTVWY"])

        # truncate to target_length if provided
        if target_length is not None and len(seq) > target_length:
            seq = seq[:target_length]
        return seq

    # convenient baseline wrapper
    def generate_baseline_sequence(self, structure: Dict) -> str:
        return self.fill_masked_positions(structure, masked_sequence=None)
