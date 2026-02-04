#!/usr/bin/env python3
"""
Calculate naturalness score for a given protein-CDS pair using GEMORNA model.

Usage:
    python calculate_naturalness.py --protein "MVLSPADKTN..." --cds "ATGGTGCTG..."
    python calculate_naturalness.py --protein_file protein.txt --cds_file cds.txt
"""

import os
import sys
import pickle
import argparse
import torch
import torch.nn.functional as F

# Add src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'src'))

from config import GEMORNA_CDS_Config
from models.gemorna_cds import Encoder, Decoder

import platform
if platform.system() == "Darwin":
    from shared.libg2m import CDS
elif platform.system() == "Linux":
    from shared.mod_xzr01 import CDS
else:
    raise RuntimeError("Unsupported OS")


def load_model(checkpoint_path=None, device=None):
    """Load GEMORNA CDS model and vocabularies."""
    if checkpoint_path is None:
        checkpoint_path = os.path.join(SCRIPT_DIR, 'checkpoints', 'gemorna_cds.pt')

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load vocabularies
    with open(os.path.join(SCRIPT_DIR, 'vocab', 'prot_vocab.pkl'), 'rb') as f:
        prot_vocab = pickle.load(f)
    with open(os.path.join(SCRIPT_DIR, 'vocab', 'cds_vocab.pkl'), 'rb') as f:
        cds_vocab = pickle.load(f)

    prot_stoi = prot_vocab.stoi if hasattr(prot_vocab, 'stoi') else prot_vocab
    cds_stoi = cds_vocab.stoi if hasattr(cds_vocab, 'stoi') else cds_vocab

    # Build model
    config = GEMORNA_CDS_Config()

    enc = Encoder(
        input_dim=config.input_dim,
        hid_dim=config.hidden_dim,
        n_layers=config.num_layers,
        n_heads=config.num_heads,
        pf_dim=config.ff_dim,
        dropout=config.dropout,
        cnn_kernel_size=config.cnn_kernel_size,
        cnn_padding=config.cnn_padding,
        device=device
    )

    dec = Decoder(
        output_dim=config.output_dim,
        hid_dim=config.hidden_dim,
        n_layers=config.num_layers,
        n_heads=config.num_heads,
        pf_dim=config.ff_dim,
        dropout=config.dropout,
        device=device
    )

    model = CDS(enc, dec, config.prot_pad_idx, config.cds_pad_idx, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    return model, prot_stoi, cds_stoi, device


def calculate_naturalness(model, protein_seq, cds_seq, prot_stoi, cds_stoi, device):
    """
    Calculate naturalness score for a protein-CDS pair.

    Args:
        model: GEMORNA CDS model
        protein_seq: Amino acid sequence (e.g., "MVLSPADKTN...")
        cds_seq: DNA/RNA coding sequence (e.g., "ATGGTGCTG...")
        prot_stoi: Protein vocabulary (string to index)
        cds_stoi: CDS vocabulary (string to index)
        device: torch device

    Returns:
        dict with 'naturalness', 'avg_log_prob', 'perplexity'
    """
    protein_seq = protein_seq.upper()
    cds_seq = cds_seq.upper().replace('U', 'T')

    # Validate
    if len(cds_seq) != len(protein_seq) * 3:
        raise ValueError(f"CDS length ({len(cds_seq)}) must be 3x protein length ({len(protein_seq)})")

    # Tokenize protein
    prot_unk = prot_stoi.get('<unk>', 0)
    prot_tokens = [prot_stoi.get(aa.lower(), prot_unk) for aa in protein_seq]
    prot_tensor = torch.tensor([prot_tokens], dtype=torch.long, device=device)

    # Tokenize CDS (lowercase codons)
    cds_seq_lower = cds_seq.lower()
    codons = [cds_seq_lower[i:i+3] for i in range(0, len(cds_seq_lower), 3)]
    cds_unk = cds_stoi.get('<unk>', 0)
    cds_tokens = [cds_stoi.get(codon, cds_unk) for codon in codons]

    # Add BOS token
    bos_idx = cds_stoi.get('<bos>', cds_stoi.get('<sos>', None))
    if bos_idx is not None:
        cds_tokens = [bos_idx] + cds_tokens

    # Prepare input/target tensors
    cds_input = torch.tensor([cds_tokens[:-1]], dtype=torch.long, device=device)
    cds_target = torch.tensor([cds_tokens[1:]], dtype=torch.long, device=device)

    with torch.no_grad():
        # Forward pass
        prot_mask = model.make_prot_mask(prot_tensor)
        cds_mask = model.make_cds_mask(cds_input)
        enc_prot = model.encoder(prot_tensor, prot_mask)
        output, _ = model.decoder(cds_input, enc_prot, cds_mask, prot_mask)

        # Get probabilities
        log_probs = F.log_softmax(output, dim=-1)
        target_log_probs = log_probs.gather(2, cds_target.unsqueeze(-1)).squeeze(-1).squeeze(0)
        probs = torch.exp(target_log_probs)

        # Compute metrics
        naturalness = torch.exp(target_log_probs.mean()).item()
        avg_log_prob = target_log_probs.mean().item()
        perplexity = torch.exp(-target_log_probs.mean()).item()

    return {
        'naturalness': naturalness,
        'avg_log_prob': avg_log_prob,
        'perplexity': perplexity
    }


def main():
    parser = argparse.ArgumentParser(description='Calculate GEMORNA naturalness score')
    parser.add_argument('--protein', type=str, help='Protein sequence')
    parser.add_argument('--cds', type=str, help='CDS sequence (DNA/RNA)')
    parser.add_argument('--protein_file', type=str, help='File containing protein sequence')
    parser.add_argument('--cds_file', type=str, help='File containing CDS sequence')
    parser.add_argument('--ckpt', type=str, default=None, help='Model checkpoint path')
    args = parser.parse_args()

    # Get sequences
    if args.protein_file:
        with open(args.protein_file) as f:
            protein_seq = f.read().strip().replace('\n', '')
    else:
        protein_seq = args.protein

    if args.cds_file:
        with open(args.cds_file) as f:
            cds_seq = f.read().strip().replace('\n', '')
    else:
        cds_seq = args.cds

    if not protein_seq or not cds_seq:
        parser.print_help()
        sys.exit(1)

    # Load model and calculate
    model, prot_stoi, cds_stoi, device = load_model(args.ckpt)
    result = calculate_naturalness(model, protein_seq, cds_seq, prot_stoi, cds_stoi, device)

    print(f"Naturalness: {result['naturalness']:.4f}")
    print(f"Avg Log Prob: {result['avg_log_prob']:.4f}")
    print(f"Perplexity: {result['perplexity']:.4f}")


if __name__ == '__main__':
    main()
