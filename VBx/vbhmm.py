#!/usr/bin/env python

# @Authors: Lukas Burget, Mireia Diez, Federico Landini, Jan Profant
# @Emails: burget@fit.vutbr.cz, mireia@fit.vutbr.cz, landini@fit.vutbr.cz, jan.profant@phonexia.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# The recipe consists in doing Agglomerative Hierachical Clustering on
# x-vectors in a first step. Then, Variational Bayes HMM over x-vectors
# is applied using the AHC output as args.initialization.
#
# A detailed analysis of this approach is presented in
# F. Landini, J. Profant, M. Diez, L. Burget
# Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization:
# theory, implementation and analysis on standard tasks
# Computer Speech & Language, 2022

import itertools
import os

import fastcluster
import h5py
import kaldi_io
import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from scipy.special import softmax

from .diarization_lib import (
    cos_similarity,
    l2_norm,
    merge_adjacent_labels,
    read_xvector_timing_dict,
    twoGMMcalib_lin,
)
from .kaldi_utils import read_plda_params_from_kaldi_format
from .VBx import VBx, VBx_plda


def write_output(fp, file_name, out_labels, starts, ends):
    for label, seg_start, seg_end in zip(out_labels, starts, ends):
        fp.write(
            f"SPEAKER {file_name} 1 {seg_start:03f} {seg_end - seg_start:03f} "
            f"<NA> <NA> {label + 1} <NA> <NA>{os.linesep}"
        )


def run_vbhmm(
    xvec_ark_file,
    segments_file,
    xvec_transform,
    plda_file,
    out_rttm_dir,
    init="AHC+VB",
    threshold=-0.015,
    lda_dim=128,
    Fa=0.3,
    Fb=17,
    loopP=0.99,
    target_energy=1.0,
    init_smoothing=5.0,
    output_2nd=False,
):
    """Run VB-HMM speaker diarization on x-vectors.

    Args:
        xvec_ark_file: path to kaldi ark file with x-vectors
        segments_file: path to segments file with x-vector timing info
        xvec_transform: path to x-vector transformation h5 file
        plda_file: path to PLDA model in Kaldi format
        out_rttm_dir: directory to write output RTTM files
        init: 'AHC' or 'AHC+VB'
        threshold: threshold (bias) for AHC
        lda_dim: LDA dimensionality for VB-HMM
        Fa: VB-HMM parameter (see VBx.VBx)
        Fb: VB-HMM parameter (see VBx.VBx)
        loopP: VB-HMM loop probability
        target_energy: PCA target energy for PLDA scoring
        init_smoothing: smoothing for AHC-to-VB initialization
        output_2nd: whether to output second-best speaker assignments
    """
    assert 0 <= loopP <= 1, f"Expecting loopP between 0 and 1, got {loopP} instead."

    segs_dict = read_xvector_timing_dict(segments_file)

    arkit = kaldi_io.read_vec_flt_ark(xvec_ark_file)
    recit = itertools.groupby(arkit, lambda e: e[0].rsplit("_", 1)[0])
    for file_name, segs in recit:
        print(file_name)
        seg_names, xvecs = zip(*segs)
        x = np.array(xvecs)

        if xvec_transform:
            with h5py.File(xvec_transform, "r") as f:
                mean1 = np.array(f["mean1"])
                mean2 = np.array(f["mean2"])
                lda = np.array(f["lda"])
                x = l2_norm(
                    lda.T.dot((l2_norm(x - mean1)).transpose()).transpose() - mean2
                )

        if not (init == "AHC" or init.endswith("VB")):
            raise ValueError("Wrong option for init.")

        if init.startswith("AHC"):
            scr_mx = cos_similarity(x)  # result is matrix
            thr, _ = twoGMMcalib_lin(scr_mx.ravel())
            scr_mx = squareform(-scr_mx, checks=False)
            lin_mat = fastcluster.linkage(
                scr_mx, method="average", preserve_input="False"
            )
            del scr_mx
            adjust = abs(lin_mat[:, 2].min())
            lin_mat[:, 2] += adjust
            labels1st = (
                fcluster(lin_mat, -(thr + threshold) + adjust, criterion="distance") - 1
            )
        if init.endswith("VB"):
            qinit = np.zeros((len(labels1st), np.max(labels1st) + 1))
            qinit[range(len(labels1st)), labels1st] = 1.0
            qinit = softmax(qinit * init_smoothing, axis=1)
            if plda_file:
                plda_params = read_plda_params_from_kaldi_format(
                    plda_file=plda_file, lda_dim=lda_dim
                )
                q, sp, L = VBx_plda(
                    x,
                    plda_params,
                    pi=qinit.shape[1],
                    gamma=qinit,
                    maxIters=40,
                    epsilon=1e-6,
                    loopProb=loopP,
                    Fa=Fa,
                    Fb=Fb,
                )
            else:
                q, sp, L = VBx(
                    x,
                    pi=qinit.shape[1],
                    gamma=qinit,
                    maxIters=40,
                    epsilon=1e-6,
                    loopProb=loopP,
                    Fa=Fa,
                    Fb=Fb,
                )

            labels1st = np.argsort(-q, axis=1)[:, 0]
            if q.shape[1] > 1:
                labels2nd = np.argsort(-q, axis=1)[:, 1]

        assert np.all(segs_dict[file_name][0] == np.array(seg_names))
        start, end = segs_dict[file_name][1].T

        starts, ends, out_labels = merge_adjacent_labels(start, end, labels1st)
        os.makedirs(out_rttm_dir, exist_ok=True)
        with open(os.path.join(out_rttm_dir, f"{file_name}.rttm"), "w") as fp:
            write_output(fp, file_name, out_labels, starts, ends)

        if output_2nd and init.endswith("VB") and q.shape[1] > 1:
            starts, ends, out_labels2 = merge_adjacent_labels(start, end, labels2nd)
            output_rttm_dir = f"{out_rttm_dir}2nd"
            os.makedirs(output_rttm_dir, exist_ok=True)
            with open(os.path.join(output_rttm_dir, f"{file_name}.rttm"), "w") as fp:
                write_output(fp, file_name, out_labels2, starts, ends)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init",
        required=True,
        type=str,
        choices=["AHC", "AHC+VB"],
        help="AHC for using only AHC or AHC+VB for VB-HMM after AHC initilization",
    )
    parser.add_argument(
        "--out-rttm-dir",
        required=True,
        type=str,
        help="Directory to store output rttm files",
    )
    parser.add_argument(
        "--xvec-ark-file", required=True, type=str, help="Kaldi ark file with x-vectors"
    )
    parser.add_argument(
        "--segments-file",
        required=True,
        type=str,
        help="File with x-vector timing info",
    )
    parser.add_argument(
        "--xvec-transform",
        required=True,
        type=str,
        help="path to x-vector transformation h5 file",
    )
    parser.add_argument(
        "--plda-file",
        required=True,
        type=str,
        help="File with PLDA model in Kaldi format",
    )
    parser.add_argument(
        "--threshold", required=True, type=float, help="threshold (bias) used for AHC"
    )
    parser.add_argument(
        "--lda-dim", required=True, type=int, help="LDA dimensionality for VB-HMM"
    )
    parser.add_argument(
        "--Fa", required=True, type=float, help="Parameter of VB-HMM (see VBx.VBx)"
    )
    parser.add_argument(
        "--Fb", required=True, type=float, help="Parameter of VB-HMM (see VBx.VBx)"
    )
    parser.add_argument(
        "--loopP", required=True, type=float, help="Parameter of VB-HMM (see VBx.VBx)"
    )
    parser.add_argument("--target-energy", required=False, type=float, default=1.0)
    parser.add_argument("--init-smoothing", required=False, type=float, default=5.0)
    parser.add_argument("--output-2nd", required=False, type=bool, default=False)

    args = parser.parse_args()

    run_vbhmm(
        xvec_ark_file=args.xvec_ark_file,
        segments_file=args.segments_file,
        xvec_transform=args.xvec_transform,
        plda_file=args.plda_file,
        out_rttm_dir=args.out_rttm_dir,
        init=args.init,
        threshold=args.threshold,
        lda_dim=args.lda_dim,
        Fa=args.Fa,
        Fb=args.Fb,
        loopP=args.loopP,
        target_energy=args.target_energy,
        init_smoothing=args.init_smoothing,
        output_2nd=args.output_2nd,
    )
