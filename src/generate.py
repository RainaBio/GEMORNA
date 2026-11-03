#!/usr/bin/env python3

import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

import pickle
import numpy as np
import pandas as pd
import torch
import sys
import argparse
import platform
if platform.system() == "Darwin":
    from shared.libg2m import *
elif platform.system() == "Linux":
    from shared.mod_xzr01 import *
else:
    raise RuntimeError("Unsupported OS")
from config import *
from models.gemorna_cds import *
from models.gemorna_utr import *

def has_noncanonical(protein_seq):
    canonical = set("ACDEFGHIKLMNPQRSTVWY*")  # 20 standard AA + stop 
    return any(residue not in canonical for residue in protein_seq.upper())

def main(args):

    mode = args.mode
    ckpt_path = args.ckpt_path
    protein_seq = args.protein_seq
    utr_len = args.utr_length
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if mode == 'cds':
        if protein_seq is None:
            raise ValueError(f'Please provide the protein sequence when using mode {mode}')

        if has_noncanonical(protein_seq):
            raise ValueError(f'The input protein sequence contains non-canonical amino acid characters.')

        prot_vocab_path = './vocab/prot_vocab.pkl'
        cds_vocab_path = './vocab/cds_vocab.pkl'

        with open(prot_vocab_path, 'rb') as f:
            prot_vocab = pickle.load(f)
        with open(cds_vocab_path, 'rb') as f:
            cds_vocab = pickle.load(f)

        model_config = GEMORNA_CDS_Config()

        enc = Encoder(
            input_dim=model_config.input_dim,
            hid_dim=model_config.hidden_dim,
            n_layers=model_config.num_layers,
            n_heads=model_config.num_heads,
            pf_dim=model_config.ff_dim,
            dropout=model_config.dropout,
            cnn_kernel_size=model_config.cnn_kernel_size,
            cnn_padding=model_config.cnn_padding,
            device=device
        )

        dec = Decoder(
            output_dim=model_config.output_dim,
            hid_dim=model_config.hidden_dim,
            n_layers=model_config.num_layers,
            n_heads=model_config.num_heads,
            pf_dim=model_config.ff_dim,
            dropout=model_config.dropout,
            device=device
        )

        model = CDS(enc, dec, model_config.prot_pad_idx, model_config.cds_pad_idx, device)

        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        model.gen(protein_seq, prot_vocab, cds_vocab, device)

    else:

        if mode == '5utr':
            model_config = GEMORNA_5UTR_Config()
            vocab = five_prime_utr_vocab
        elif mode == '3utr':
            model_config = GEMORNA_3UTR_Config()
            vocab = three_prime_utr_vocab
        else:
            print("Wrong mode!")
            sys.exit()

        model = UTR(model_config)
        model.load_state_dict(torch.load(ckpt_path, map_location=device)['model'])
        model.to(device)
        model.eval()
        model.gen(mode, vocab, device, utr_len)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("This is script to pretrain GEMORNA CDS model")
    parser.add_argument("--mode", type=str, help="cds, 5utr or 3utr")
    parser.add_argument("--ckpt_path", type=str, help="Path to load pre-trained models")
    parser.add_argument("--protein_seq", type=str, help="Specify the input protein sequence for generating CDS")
    parser.add_argument("--utr_length", type=str, help="Specify the length (short, medium, long) of UTR sequences")
    args = parser.parse_args()
    main(args)


