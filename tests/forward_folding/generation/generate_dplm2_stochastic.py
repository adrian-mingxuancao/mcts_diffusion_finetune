#!/usr/bin/env python3
"""
Generate DPLM-2 sequences using STOCHASTIC sampling (matching MCTS)
This provides a fair comparison by using the same generation strategy
"""

import argparse
import os

HF_SCRATCH_CACHE = "/net/scratch/caom/.cache/huggingface"
if "HF_HOME" not in os.environ and os.path.isdir(HF_SCRATCH_CACHE):
    os.environ["HF_HOME"] = HF_SCRATCH_CACHE
if "TRANSFORMERS_CACHE" not in os.environ and "HF_HOME" in os.environ:
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["HF_HOME"], "transformers")
if "HUGGINGFACE_HUB_CACHE" not in os.environ and "HF_HOME" in os.environ:
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HOME"]

# Patch ESM
def patch_esm_regression_weights():
    try:
        import esm.pretrained as _esm_pkg
        def skip_regression_weights(model_name):
            return False
        _esm_pkg._has_regression_weights = skip_regression_weights
        print("✓ ESM regression weights patched")
    except ImportError:
        print("⚠ ESM not available for patching")

patch_esm_regression_weights()

import torch
import tree
from Bio import SeqIO
from peft.peft_model import PeftModel
from tqdm import tqdm

from byprot.models.dplm2 import DPLM2Bit
from byprot.models.dplm2 import (
    MultimodalDiffusionProteinLanguageModel as DPLM2,
)


def ensure_dplm2_tokenizer_registered():
    try:
        from byprot.datamodules.dataset.tokenized_protein import DPLM2Tokenizer
        from transformers import AutoTokenizer
    except ImportError as exc:
        print(f"⚠️ Unable to register DPLM2Tokenizer ({exc})")
        return

    try:
        DPLM2Tokenizer.register_for_auto_class("AutoTokenizer")
    except ValueError:
        pass
    try:
        AutoTokenizer.register("DPLM2Tokenizer", DPLM2Tokenizer)
    except ValueError:
        pass

    globals()["DPLM2Tokenizer"] = DPLM2Tokenizer


ensure_dplm2_tokenizer_registered()


def initialize_conditional_generation(
    fasta_path, tokenizer, device, args, model=None
):
    input_data_aatype = []
    input_data_struct_tokens = []
    input_data_name = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        input_data_name.append(record.name)
        if args.task == "folding":
            # Forward folding: sequence -> structure
            aatype = str(record.seq)
            aatype = tokenizer.aa_cls_token + aatype + tokenizer.aa_eos_token
            struct_tokens = tokenizer.struct_mask_token * len(record.seq)
            struct_tokens = (
                tokenizer.struct_cls_token
                + struct_tokens
                + tokenizer.struct_eos_token
            )
        elif args.task == "inverse_folding":
            # Inverse folding: structure -> sequence
            aatype = tokenizer.aa_mask_token * len(record.seq.split(","))
            aatype = tokenizer.aa_cls_token + aatype + tokenizer.aa_eos_token
            struct_tokens = "".join(str(record.seq).split(","))
            struct_tokens = (
                tokenizer.struct_cls_token
                + struct_tokens
                + tokenizer.struct_eos_token
            )
        else:
            raise NotImplementedError(f"Task {args.task} not supported")
        input_data_aatype.append(aatype)
        input_data_struct_tokens.append(struct_tokens)

    # sorted by length
    len_input = [len(seq) for seq in input_data_aatype]
    sorted_batch = sorted(
        zip(input_data_name, input_data_aatype, input_data_struct_tokens, len_input),
        key=lambda x: x[3],
    )
    input_data_name, input_data_aatype, input_data_struct_tokens, len_input = zip(
        *sorted_batch
    )

    return input_data_aatype, input_data_struct_tokens, input_data_name


def build_batch(input_data_aatype, input_data_struct_tokens, input_data_name, 
                tokenizer, model, device, args):
    """Build batch following the original DPLM-2 approach"""
    batch_struct = tokenizer.batch_encode_plus(
        input_data_struct_tokens,
        add_special_tokens=False,
        padding="longest",
        return_tensors="pt",
    )

    batch_aa = tokenizer.batch_encode_plus(
        input_data_aatype,
        add_special_tokens=False,
        padding="longest",
        return_tensors="pt",
    )

    input_tokens = torch.concat(
        [batch_struct["input_ids"], batch_aa["input_ids"]], dim=1
    )
    input_tokens = input_tokens.to(device)

    aa_type = 1
    struct_type = 0
    non_special = model.get_non_special_symbol_mask(input_tokens)
    type_ids = model.get_modality_type(input_tokens)

    # Task-specific masking
    if args.task == "inverse_folding":
        # Inverse folding: mask aa tokens, keep structure tokens
        input_tokens.masked_fill_(
            (type_ids == aa_type) & non_special,
            tokenizer._token_to_id[tokenizer.aa_mask_token],
        )
        mask_type = struct_type  # Don't unmask struct tokens
    elif args.task == "folding":
        # Forward folding: mask structure tokens, keep aa tokens
        input_tokens.masked_fill_(
            (type_ids == struct_type) & non_special,
            tokenizer._token_to_id[tokenizer.struct_mask_token],
        )
        mask_type = aa_type  # Don't unmask aa tokens
    
    # Set partial_mask to indicate which tokens should NOT be unmasked
    partial_mask = type_ids == mask_type
    
    return {
        "input_tokens": input_tokens,
        "partial_mask": partial_mask,
        "name": input_data_name,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                       help="Model name, e.g., airkingbd/dplm2_650m")
    parser.add_argument("--task", type=str, default="inverse_folding", 
                       choices=["folding", "inverse_folding"])
    parser.add_argument("--input_fasta_path", type=str, required=True)
    parser.add_argument("--saveto", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_iter", type=int, default=100)
    
    # STOCHASTIC SAMPLING (matching MCTS)
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for stochastic sampling")
    parser.add_argument("--unmasking_strategy", type=str, default="stochastic1.0",
                       help="Unmasking strategy (use stochastic{temp} to match MCTS)")
    parser.add_argument("--sampling_strategy", type=str, default="annealing@2.2:1.0",
                       help="Sampling strategy (use annealing@2.2:1.0 to match MCTS)")
    
    # Random seed for reproducibility
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed (None for random)")

    args = parser.parse_args()
    
    # Set random seed if specified
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"🎲 Random seed set to: {args.seed}")
    else:
        print("🎲 Using random seed (not fixed)")
    
    print(f"\n🔧 Generation Settings (STOCHASTIC - matching MCTS):")
    print(f"   Unmasking: {args.unmasking_strategy}")
    print(f"   Sampling:  {args.sampling_strategy}")
    print(f"   Temperature: {args.temperature}")
    print(f"   Max iterations: {args.max_iter}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model: {args.model_name}")
    model = DPLM2.from_pretrained(args.model_name).to(device).eval()
    tokenizer = model.tokenizer

    # Initialize data
    input_data_aatype, input_data_struct_tokens, input_data_name = initialize_conditional_generation(
        args.input_fasta_path, tokenizer, device, args, model
    )

    # Generate
    os.makedirs(args.saveto, exist_ok=True)
    output_dir = os.path.join(args.saveto, args.task)
    os.makedirs(output_dir, exist_ok=True)

    # Process in batches
    batch_size = args.batch_size
    num_batches = (len(input_data_name) + batch_size - 1) // batch_size
    
    print(f"\nGenerating {len(input_data_name)} sequences in {num_batches} batches...")
    
    for i in tqdm(range(0, len(input_data_name), batch_size)):
        batch_aatype = input_data_aatype[i:i+batch_size]
        batch_struct = input_data_struct_tokens[i:i+batch_size]
        batch_names = input_data_name[i:i+batch_size]
        
        # Build batch
        batch = build_batch(batch_aatype, batch_struct, batch_names, 
                           tokenizer, model, device, args)
        
        # Generate (use same autocast as deterministic script)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model.generate(
                input_tokens=batch["input_tokens"],
                max_iter=args.max_iter,
                temperature=args.temperature,
                unmasking_strategy=args.unmasking_strategy,
                sampling_strategy=args.sampling_strategy,
                partial_masks=batch["partial_mask"],
            )

        # Extract generated tokens
        generated_tokens = outputs["output_tokens"]
        
        # Save outputs
        for j, name in enumerate(batch["name"]):
            tokens = generated_tokens[j]
            decoded_seq = tokenizer.decode(tokens.cpu().tolist())
            
            # Task-specific output format (matching generate_dplm2_patched_v2.py)
            if args.task == "inverse_folding":
                # For inverse folding: extract only AA sequence
                # Structure tokens come first, then AA tokens
                aa_type = 1
                type_ids = model.get_modality_type(tokens.unsqueeze(0))
                aa_positions = (type_ids[0] == aa_type).nonzero(as_tuple=False).flatten()
                
                if len(aa_positions) > 0:
                    aa_tokens = tokens[aa_positions].cpu().tolist()
                    sequence = tokenizer.decode(aa_tokens)
                    # Remove special tokens
                    sequence = sequence.replace(tokenizer.aa_cls_token, "")
                    sequence = sequence.replace(tokenizer.aa_eos_token, "")
                else:
                    sequence = decoded_seq
            else:
                # For folding: save full sequence (structure tokens + AA tokens)
                sequence = decoded_seq
            
            # Write FASTA file
            output_path = os.path.join(output_dir, f"{name}.fasta")
            with open(output_path, "w") as f:
                f.write(f">{name}\n{sequence}\n")

    print(f"\n✅ Generation complete! Results saved to: {output_dir}")
    print(f"\n📊 Settings used:")
    print(f"   Model: {args.model_name}")
    print(f"   Unmasking: {args.unmasking_strategy}")
    print(f"   Sampling: {args.sampling_strategy}")
    print(f"   Temperature: {args.temperature}")
    print(f"   Seed: {args.seed if args.seed is not None else 'random'}")


if __name__ == "__main__":
    main()
