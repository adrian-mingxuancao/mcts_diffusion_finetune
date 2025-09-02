import os
import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

# Import the working version from the GitHub repository
from core.dplm2_integration_fixed import DPLM2Integration

class DPLM2IntegrationCorrected(DPLM2Integration):
    """
    Corrected version that uses ID path for structure (robust) and text path for AA.
    This matches the working generate_dplm2_patched_v2.py logic exactly.
    """
    
    def _create_dplm2_batch(self, structure: Dict, target_length: int, masked_sequence: str = None) -> Dict:
        """
        Create DPLM-2 batch for generation following the working version exactly.
        Uses ID path for structure (robust) and text path for AA (as before).
        """
        tok = self.model.tokenizer

        # ---------- 1) STRUCTURE → IDS (not text) ----------
        # Prefer pretokenized data; else fallback to coordinates / FASTA as you already do
        struct_ids, Ls = None, None
        if structure.get("struct_seq") is not None or structure.get("struct_ids") is not None:
            # Use the robust helper that adds BOS/EOS and validates range
            struct_ids, Ls = self._tokens_to_struct_ids(structure)   # [1, Ls] LongTensor on model device
        else:
            # Your existing coordinate/FASTA fallbacks stay as-is, but MUST end in struct_ids, Ls
            try:
                struct_ids, Ls = self._coordinates_to_structure_tokens(structure)
            except Exception as coord_error:
                # your FASTA fallback here; at the end, call _tokens_to_struct_ids again
                pdb_id = structure.get('pdb_id', '')
                chain_id = structure.get('chain_id', '')
                structure_name = f"{pdb_id}_{chain_id}" if pdb_id and chain_id else structure.get('name', '').replace('CAMEO ', '')
                from utils.struct_loader import load_struct_seq_from_fasta
                struct_seq = load_struct_seq_from_fasta("/home/caom/AID3/dplm/data-bin/cameo2022/struct.fasta", structure_name)
                structure = dict(structure)
                structure['struct_seq'] = struct_seq
                struct_ids, Ls = self._tokens_to_struct_ids(structure)

        # ---------- 2) AA → TEXT → IDS (keep your existing AA text path) ----------
        if masked_sequence is None:
            # full inverse folding baseline: AA body all mask tokens of length Ls
            aa_body = tok.aa_mask_token * Ls
        else:
            ms = str(masked_sequence)
            aa_body = "".join(tok.aa_mask_token if c == "X" else c for c in ms)

        aa_text = tok.aa_cls_token + aa_body + tok.aa_eos_token

        batch_aa = tok.batch_encode_plus(
            [aa_text],
            add_special_tokens=False,
            padding=False,            # one sample
            truncation=False,         # avoid HF warning
            return_tensors="pt"
        )
        aa_ids = batch_aa["input_ids"].to(self.device)              # [1, La]
        La = aa_ids.shape[1]

        # ---------- 3) CONCAT STRUCT + AA IDS ----------
        input_tokens = torch.cat([struct_ids.to(self.device), aa_ids], dim=1)  # [1, Ls+La]

        # ---------- 4) TYPE IDS / SPECIALS ----------
        type_ids = self.model.get_modality_type(input_tokens)
        non_special = self.model.get_non_special_symbol_mask(input_tokens)

        # ---------- 5) PARTIAL MASK (don't freeze AA mask token) ----------
        aa_mask_id = self._get_token_id(tok.aa_mask_token)
        struct_type, aa_type = 0, 1
        special = ~non_special
        is_mask_token = (input_tokens == aa_mask_id)

        partial_mask = (type_ids == struct_type) | (special & ~is_mask_token)

        # If baseline, ensure AA body are mask tokens (already true from aa_text path)
        if masked_sequence is not None:
            for i, c in enumerate(ms):
                pos = Ls + 1 + i   # AA-CLS at Ls, body starts at Ls+1
                partial_mask[0, pos] = (c != "X")   # unfreeze X's

        # Debug logs
        aa_body_count = int((type_ids[0] == aa_type).sum().item()) - 2
        trainable = int((~partial_mask).sum().item())
        print(f"   🧮 AA body to diffuse: {aa_body_count}")
        print(f"   ✅ Trainable AA positions this step: {trainable}")
        print(f"   📊 Batch info: input_tokens={input_tokens.shape}, partial_mask={partial_mask.shape}")

        return {"input_tokens": input_tokens, "partial_mask": partial_mask}
