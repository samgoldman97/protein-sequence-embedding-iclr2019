"""A module to help load and use the pretrained embeddings. 

Because the pretrained embeddings are created with an older version of torch, a
new model is created with the same parameters and the state dict is
transferred into the new model. This allows these embeddings to be more easily
used as a python package. 

"""

import os
import shutil 
import urllib.request as request
import logging
from contextlib import closing

from torch import nn
import torch
import numpy as np 
from typing import Optional, List
from tqdm import tqdm

# Use absolute imports
from bepler_embedding import alphabets
from bepler_embedding import utils
from bepler_embedding.models import multitask, embedding, sequence

# Get default cache location
# Taken from TAPE protein repository
try:
    from torch.hub import _get_torch_home
    torch_cache_home = _get_torch_home()
except ImportError:
    torch_cache_home = os.path.expanduser(
        os.getenv('TORCH_HOME', os.path.join(
            os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))

DEFAULT_CACHE_PATH = os.path.join(torch_cache_home, 'bepler_models_cache')
MODEL_LINK  = "http://bergerlab-downloads.csail.mit.edu/bepler-protein-sequence-embeddings-from-structure-iclr2019/pretrained_models.tar.gz"

# Mirror containing just the state dict
MODEL_MIRROR_LINK= "https://www.dropbox.com/s/kvebr0c038z9bhm/bepler_SCOPCM_state_dict.pk?dl=0"

# Make alphabet
uniprot_alphabet = alphabets.Uniprot21()

def download_ftp(ftp_link: str, outfile: str):
    """download_ftp.

    Args:
        ftp_link (str): ftp_link
        outfile (str): outfile
    """

    with closing(request.urlopen(ftp_link)) as r:
        with open(outfile, 'wb') as f:
            shutil.copyfileobj(r, f)

#def load_pretrained_embedding(in_file : Optional[str] = None) -> nn.Module: 
#    """ load_pretrained_embedding.
#
#    Load the pretrained embedding. If it doesn't exist or no in file is
#    passed, download it.
#
#
#    Args:
#        in_file (Optional[str]) : Optional input argument
#
#    Return: 
#        nn.Module
#    pretrained_models/ssa_L1_100d_lstm3x512_lm_i512_mb64_tau0.5_lambda0.1_p0.05_epoch100.sav
#    """
#    # Try to get from cache path
#    model_name = "pretrained_models/ssa_L1_100d_lstm3x512_lm_i512_mb64_tau0.5_lambda0.1_p0.05_epoch100.sav" 
#    if in_file is None or not os.path.isfile(in_file): 
#        pretrained_loc = os.path.join(DEFAULT_CACHE_PATH, model_name)
#        # If the path doesn't exist, downlaod it
#        if not os.path.isfile(pretrained_loc): 
#
#            # Make the directory if it doesn't exist
#            if not os.path.isdir(DEFAULT_CACHE_PATH): 
#                os.makedirs(DEFAULT_CACHE_PATH, exist_ok=True)
#
#            # Download the zipped file first
#            temp_file = os.path.join(DEFAULT_CACHE_PATH, "temp.tar.gz")
#            logging.info(f"Unable to find saved models, downloading to {temp_file}")
#            download_ftp(MODEL_LINK, temp_file)
#            logging.info(f"Done downloading. Unzipping file into {DEFAULT_CACHE_PATH}")
#            # Unpack zip
#            shutil.unpack_archive(temp_file, DEFAULT_CACHE_PATH)
#            logging.info(f"Unzip complete. Removing {temp_file}")
#            os.remove(temp_file)
#        if not os.path.isfile(pretrained_loc): 
#            raise ValueError(f"After download, still unable to load {pretrained_loc}")
#    else: 
#        pretrained_loc = in_file
#
#    # Now try to load the model
#
#    return torch.load(pretrained_loc)

def load_pretrained_embedding(in_file : Optional[str] = None) -> nn.Module: 
    """ load_pretrained_embedding.

    Load the pretrained embedding. If it doesn't exist or no in file is
    passed, download it.


    Args:
        in_file (Optional[str]) : Optional input argument

    Return: 
        nn.Module
    pretrained_models/ssa_L1_100d_lstm3x512_lm_i512_mb64_tau0.5_lambda0.1_p0.05_epoch100.sav
    """
    # Try to get from cache path
    model_name = "bepler_SCOPCM_state_dict.pk" 
    if in_file is None or not os.path.isfile(in_file): 
        pretrained_loc = os.path.join(DEFAULT_CACHE_PATH, model_name)
        # If the path doesn't exist, downlaod it
        if not os.path.isfile(pretrained_loc): 
            # Make the directory if it doesn't exist
            if not os.path.isdir(DEFAULT_CACHE_PATH): 
                os.makedirs(DEFAULT_CACHE_PATH, exist_ok=True)

            # Download the zipped file first
            logging.info(f"Unable to find saved models, downloading to {model_name}")
            download_ftp(MODEL_MIRROR_LINK, model_name)
            logging.info(f"Done downloading.") 
            # Unpack zip
            #shutil.unpack_archive(temp_file, DEFAULT_CACHE_PATH)
            #logging.info(f"Unzip complete. Removing {temp_file}")
            #os.remove(temp_file)
        if not os.path.isfile(pretrained_loc): 
            raise ValueError(f"After download, still unable to load {pretrained_loc}")
    else: 
        pretrained_loc = in_file
    # Now try to load the model
    return torch.load(pretrained_loc)

def remake_pretrained_model(state_dict : dict) -> nn.Module: 
    """ remake_pretrained_model.

    This function hard codes the parameters used for the pretrained bepler
    embedding model and reconstructs the model uploaded with the paper with the
    goal of loading the state dict into the most recent version of pytorch. 

    Args: 
        stat_dict: This should be an ODict containing the embeddinig state dict 
    Return: 
        nn.Module

    """
    # First make BiLM 
    ntokens = len(uniprot_alphabet)
    nin = ntokens + 1
    nout = ntokens
    embedding_dim = 21
    hidden_dim = 1024
    num_layers = 2
    mask_idx = ntokens
    dropout = 0
    tied=True
    bilm = sequence.BiLM(nin, nout, embedding_dim, hidden_dim, num_layers,
                         mask_idx=mask_idx, dropout=dropout, tied=tied)

    # Then make stacked RNN 
    embedding_module = embedding.StackedRNN(ntokens, 512, 512, 100, 
                                            nlayers=3, dropout=0, lm=bilm)

    embedding_module.load_state_dict(state_dict)
    new_model = embedding_module
    return new_model

def get_default_saved_model() -> nn.Module: 
    """ get_default_saved_model.

    Return: 
        nn.Module: Default embedding module. This is of type stacked_RNN 
    """
    old_model = load_pretrained_embedding()
    #new_model = remake_pretrained_model(old_model.embedding.state_dict())
    new_model = remake_pretrained_model(old_model)
    return new_model


def encode_seqs(seqs : List[str]) -> torch.Tensor: 
    """encode_seqs.

    Args:
        seqs (List[srt]): seqs

    Returns:
        torch.Tensor:
    """
    encoded_seqs = [torch.from_numpy(np.array(uniprot_alphabet.encode(i.encode()))).long() for i in seqs]
    return encoded_seqs
    
def embed_sequences(model : nn.Module, seqs : List[str], 
                    batch_size : int = 32, 
                    gpu : bool = False) -> np.ndarray: 
    """embed_sequences.

    Args:
        model (nn.Module): model
        seqs (List[str]): seqs
        batch_size (int): batch_size
        gpu (bool): gpu

    Returns:
        np.ndarray:
    """

    encoded_seqs = encode_seqs(seqs)
    identity_collate = lambda x : x 
    loader = torch.utils.data.DataLoader(encoded_seqs, batch_size=batch_size,
                                         collate_fn=identity_collate)
    if gpu: 
        model = model.cuda()
    return_seqs = []
    with torch.no_grad(): 
        model.eval()
        # 
        for batch in tqdm(loader):
            X, order = utils.pack_sequences(encoded_seqs)
            if gpu: 
                X = X.cuda()

            out = model.forward(X)
            unpacked = utils.unpack_sequences(out, order)
            return_seqs.extend([i.detach().cpu().numpy() for i in unpacked])

    return return_seqs
