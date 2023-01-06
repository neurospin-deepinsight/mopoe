# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define the different workflows used during the analysis.
"""

# Imports
import os
import json
from types import SimpleNamespace
import torch
from mopoe.run_epochs import run_epochs
from mopoe.utils.filehandling import create_dir_structure
from mopoe.multimodal_cohort.experiment import MultimodalExperiment
from mopoe.color_utils import print_title


def train_exp(dataset, datasetdir, outdir, input_dims, latent_dim=20,
              num_hidden_layers=1, allow_missing_blocks=False, beta=5.,
              likelihood="normal", initial_learning_rate=0.002, batch_size=256,
              n_epochs=2500, eval_freq=25, eval_freq_fid=100,
              data_multiplications=1, dropout_rate=0., initial_out_logvar=-3.,
              learn_output_scale=False):
    """ Train the model.

    Parameters
    ----------
    dataset: str
        the dataset name: euaims or hbn.
    datasetdir: str
        the path to the dataset associated data.
    outdir: str
        the destination folder.
    input_dims: list of int
        input dimension for each modality.
    latent_dim: int, default 20
        dimension of common factor latent space.
    num_hidden_layers: int, default 1
        number of hidden laters in the model.
    allow_missing_blocks: bool, default False
        optionally, allows for missing modalities.
    beta: float, default 5
        default weight of sum of weighted divergence terms.
    likelihood: str, default 'normal'
        output distribution.
    initial_learning_rate: float, default 0.002
        starting learning rate.
    batch_size: int, default 256
        batch size for training.
    n_epochs: int, default 2500
        the number of epochs for training.
    eval_freq: int, default 25
        frequency of evaluation of latent representation of generative
        performance (in number of epochs).
    eval_freq_fid: int, default 100
        frequency of evaluation of latent representation of generative
        performance (in number of epochs).
    data_multiplications: int, default 1
        number of pairs per sample.
    dropout_rate: float, default 0
        the dropout rate in the training.
    initial_out_logvar: float, default -3
        initial output logvar.
    learn_output_scale: bool, default False
        optionally, allows for different scales per feature.
    """
    print_title(f"TRAIN: {dataset}")
    flags = SimpleNamespace(
        dataset=dataset, datasetdir=datasetdir, dropout_rate=dropout_rate,
        allow_missing_blocks=allow_missing_blocks, batch_size=batch_size,
        beta=beta, beta_1=0.9, beta_2=0.999, beta_content=1.0,
        beta_style=1.0, calc_nll=False, calc_prd=False,
        class_dim=latent_dim, data_multiplications=data_multiplications,
        dim=64, dir_data="../data", dir_experiment=outdir, dir_fid=None,
        div_weight=None, div_weight_uniform_content=None,
        end_epoch=n_epochs, eval_freq=eval_freq, eval_freq_fid=eval_freq_fid,
        factorized_representation=False, img_size_m1=28, img_size_m2=32,
        inception_state_dict="../inception_state_dict.pth",
        initial_learning_rate=initial_learning_rate,
        initial_out_logvar=initial_out_logvar, input_dim=input_dims,
        joint_elbo=False, kl_annealing=0, include_prior_expert=False,
        learn_output_scale=learn_output_scale, len_sequence=8,
        likelihood=likelihood, load_saved=False, method='joint_elbo',
        mm_vae_save="mm_vae", modality_jsd=False, modality_moe=False,
        modality_poe=False, num_channels_m1=1, num_channels_m2=3,
        num_classes=2, num_hidden_layers=num_hidden_layers,
        num_samples_fid=10000, num_training_samples_lr=500,
        poe_unimodal_elbos=True, save_figure=False, start_epoch=0, style_dim=0,
        subsampled_reconstruction=True)
    print(flags)
    use_cuda = torch.cuda.is_available()
    flags.device = torch.device("cuda" if use_cuda else "cpu")
    if flags.method == "poe":
        flags.modality_poe = True
        flags.poe_unimodal_elbos = True
    elif flags.method == "moe":
        flags.modality_moe = True
    elif flags.method == "jsd":
        flags.modality_jsd = True
    elif flags.method == "joint_elbo":
        flags.joint_elbo = True
    else:
        print("Method not implemented...exit!")
        return

    flags.num_mods = len(flags.input_dim)
    if flags.div_weight_uniform_content is None:
        flags.div_weight_uniform_content = 1 / (flags.num_mods + 1)
    flags.alpha_modalities = [flags.div_weight_uniform_content]
    if flags.div_weight is None:
        flags.div_weight = 1 / (flags.num_mods + 1)
    flags.alpha_modalities.extend([
        flags.div_weight for _ in range(flags.num_mods)])
    create_dir_structure(flags)

    alphabet_path = os.path.join(os.getcwd(), "alphabet.json")
    with open(alphabet_path) as alphabet_file:
        alphabet = str("".join(json.load(alphabet_file)))
    mst = MultimodalExperiment(flags, alphabet)
    mst.set_optimizer()
    run_epochs(mst)
