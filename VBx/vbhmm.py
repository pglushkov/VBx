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
import logging
import os
from dataclasses import dataclass
from pathlib import Path

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
from .kaldi_utils import PLDAParams, read_plda_params_from_kaldi_format
from .VBx import VBx, VBx_plda

log_level = os.environ.get("LOG_LEVEL", "INFO")
print(f" === using log-level: {log_level}")
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HumanReadableDataFromKaldi:
    filename: str
    xvec_labels: tuple[str]
    xvecs: np.ndarray
    time_labels: tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class LDAParams:
    mean1: np.ndarray
    mean2: np.ndarray
    lda: np.ndarray


def debug_trace_array(arr: np.ndarray, label: str):
    mean = np.mean(arr)
    std = np.std(arr)
    max = np.max(arr)
    min = np.min(arr)
    logger.debug(
        f"  tracing arr {label}: shape={arr.shape}, mean={mean}, std={std}, max={max}, min={min}"
    )


def write_output(fp, file_name, out_labels, starts, ends):
    for label, seg_start, seg_end in zip(out_labels, starts, ends):
        fp.write(
            f"SPEAKER {file_name} 1 {seg_start:03f} {seg_end - seg_start:03f} "
            f"<NA> <NA> {label + 1} <NA> <NA>{os.linesep}"
        )


def get_clusters_from_xvectors(
    x_vecs: np.ndarray, threshold: float, use_cpp: bool = False
):
    if use_cpp:
        import vbx_native

        N = x_vecs.shape[0]
        sim_matrix = vbx_native.cosine_similarity(x_vecs, condense_result=False)
        thr = vbx_native.ahc_threshold(sim_matrix.ravel())
        sim_condensed = squareform(-sim_matrix, checks=False)
        linkage = vbx_native.average_linkage(sim_condensed, N)
        adjust = abs(linkage[:, 2].min())
        linkage[:, 2] += adjust
        labels1st = (
            vbx_native.fcluster_distance(linkage, -(thr + threshold) + adjust) - 1
        )
    else:
        scr_mx = cos_similarity(x_vecs)  # result is matrix
        thr, _ = twoGMMcalib_lin(scr_mx.ravel())
        scr_mx = squareform(-scr_mx, checks=False)
        lin_mat = fastcluster.linkage(scr_mx, method="average", preserve_input="False")
        del scr_mx
        adjust = abs(lin_mat[:, 2].min())
        lin_mat[:, 2] += adjust
        labels1st = (
            fcluster(lin_mat, -(thr + threshold) + adjust, criterion="distance") - 1
        )
    return labels1st


def vbhmm_diarization_from_clusters(
    xvecs: np.ndarray,
    xvecs_time_labels: tuple[np.ndarray, np.ndarray],
    plda_params: PLDAParams | None,
    lda_params: LDAParams | None,
    qinit: np.ndarray,
    loop_prob: float,
    Fa: float,
    Fb: float,
    max_iters: int = 40,
    epsilon: float = 1e-6,
    use_cpp: bool = False,
    output_2nd: bool = False,
):
    if lda_params:  # TODO: move it to C++
        lda = lda_params.lda
        mean1 = lda_params.mean1
        mean2 = lda_params.mean2
        xvecs = l2_norm(l2_norm(xvecs - mean1) @ lda - mean2)

    debug_trace_array(xvecs, "xvecs")
    debug_trace_array(qinit, "initial clusters(gamma)")

    if use_cpp:
        assert plda_params is not None, (
            "C++ version currently requires PLDA to be provided"
        )
        import vbx_native

        q, sp, elbo = vbx_native.vbhmm(
            xvecs,
            qinit,
            plda_params.mean,
            plda_params.transform,
            plda_params.psi,
            loop_prob=loop_prob,
            Fa=Fa,
            Fb=Fb,
            max_iters=max_iters,
            epsilon=epsilon,
        )
    else:
        if plda_params is not None:
            q, sp, L = VBx_plda(
                xvecs,
                plda_params,
                pi=qinit.shape[1],
                gamma=qinit,
                maxIters=max_iters,
                epsilon=epsilon,
                loopProb=loop_prob,
                Fa=Fa,
                Fb=Fb,
            )
        else:
            q, sp, L = VBx(
                xvecs,
                pi=qinit.shape[1],
                gamma=qinit,
                maxIters=max_iters,
                epsilon=epsilon,
                loopProb=loop_prob,
                Fa=Fa,
                Fb=Fb,
            )

    labels1st = np.argsort(-q, axis=1)[:, 0]
    if q.shape[1] > 1 and output_2nd:
        labels2nd = np.argsort(-q, axis=1)[:, 1]
    else:
        labels2nd = None

    # logger.debug(f"raw diarization results: q_shape={q.shape}, {q}")
    # logger.debug(f"assigned clusters: shape={labels1st.shape}, labels1st={labels1st}")
    debug_trace_array(q, "raw_diar_results(q)")
    debug_trace_array(labels1st, "assigned clusters(labels1st)")

    in_starts, in_ends = xvecs_time_labels

    starts, ends, out_labels = merge_adjacent_labels(in_starts, in_ends, labels1st)

    return (starts, ends, out_labels, labels2nd)


def deal_with_kaldi_bs(
    xvec_ark_file: str | Path,
    segments_file: str | Path,
    lda_transform: str | Path | None,
    lda_dim: int,
    plda_file: str | Path | None,
) -> tuple[HumanReadableDataFromKaldi, PLDAParams | None, LDAParams | None]:
    arkit = kaldi_io.read_vec_flt_ark(xvec_ark_file)
    recit = itertools.groupby(arkit, lambda e: e[0].rsplit("_", 1)[0])
    segs_dict = read_xvector_timing_dict(segments_file)

    (file_name, segs) = next(recit)
    logger.info(f"Filename from kaldi file: {file_name}")
    seg_names, xvecs = zip(*segs)
    # x =

    if lda_transform:
        with h5py.File(lda_transform, "r") as f:
            mean1 = np.array(f["mean1"])
            mean2 = np.array(f["mean2"])
            lda = np.array(f["lda"])
            lda_params = LDAParams(mean1=mean1, mean2=mean2, lda=lda)
    else:
        lda_params = None

    if plda_file:
        if os.path.isfile(plda_file):
            # assume its kaldi plda-file
            plda_params = read_plda_params_from_kaldi_format(
                plda_file=plda_file, lda_dim=lda_dim
            )
        elif os.path.isdir(plda_file):
            # assume its directory with plda params saved separately
            mean = np.load(os.path.join(plda_file, "mean.npy")).squeeze()
            transform = np.load(os.path.join(plda_file, "transform.npy"))
            psi = np.load(os.path.join(plda_file, "psi.npy")).squeeze()
            plda_dim = psi.shape[0]
            plda_params = PLDAParams(
                dim=plda_dim, mean=mean, transform=transform, psi=psi
            )
    else:
        plda_params = None

    # this magnificent piece of BS was done inside the file-processing loop
    assert np.all(segs_dict[file_name][0] == np.array(seg_names))
    in_starts, in_ends = segs_dict[file_name][1].T

    data_from_kaldi_file = HumanReadableDataFromKaldi(
        filename=file_name,
        xvec_labels=seg_names,
        xvecs=np.ascontiguousarray(np.array(xvecs)),
        time_labels=(in_starts, in_ends),
    )

    return (data_from_kaldi_file, plda_params, lda_params)


def run_vbhmm(
    xvec_ark_file,
    segments_file,
    xvec_transform,
    plda_file,
    out_rttm_dir,
    threshold=-0.015,
    lda_dim=128,
    Fa=0.3,
    Fb=17.0,
    loopP=0.99,
    target_energy=1.0,
    init_smoothing=5.0,
    output_2nd=False,
    use_cpp=False,
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

    data_from_kaldi, plda_params, lda_params = deal_with_kaldi_bs(
        xvec_ark_file,
        segments_file,
        xvec_transform,
        lda_dim,
        plda_file,
    )

    xvecs = data_from_kaldi.xvecs
    file_name = data_from_kaldi.filename
    segment_starts = data_from_kaldi.time_labels[0]
    segment_ends = data_from_kaldi.time_labels[1]

    # get initial clusters structure
    labels1st = get_clusters_from_xvectors(xvecs, threshold, use_cpp=use_cpp)

    # form initial class labels from clusterization results
    qinit = np.zeros((len(labels1st), np.max(labels1st) + 1))
    qinit[range(len(labels1st)), labels1st] = 1.0
    qinit = softmax(qinit * init_smoothing, axis=1)

    merged_starts, merged_ends, out_labels, labels2nd = vbhmm_diarization_from_clusters(
        xvecs=xvecs,
        xvecs_time_labels=data_from_kaldi.time_labels,
        plda_params=plda_params,
        lda_params=lda_params,
        qinit=qinit,
        loop_prob=loopP,
        Fa=Fa,
        Fb=Fb,
        max_iters=40,
        epsilon=1e-6,
        use_cpp=use_cpp,
        output_2nd=False,
    )

    os.makedirs(out_rttm_dir, exist_ok=True)
    with open(os.path.join(out_rttm_dir, f"{file_name}.rttm"), "w") as fp:
        write_output(fp, file_name, out_labels, merged_starts, merged_ends)

    if output_2nd:
        merged_starts, merged_ends, out_labels2 = merge_adjacent_labels(
            segment_starts, segment_ends, labels2nd
        )
        output_rttm_dir = f"{out_rttm_dir}2nd"
        os.makedirs(output_rttm_dir, exist_ok=True)
        with open(os.path.join(output_rttm_dir, f"{file_name}.rttm"), "w") as fp:
            write_output(fp, file_name, out_labels2, merged_starts, merged_ends)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
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
        threshold=args.threshold,
        lda_dim=args.lda_dim,
        Fa=args.Fa,
        Fb=args.Fb,
        loopP=args.loopP,
        target_energy=args.target_energy,
        init_smoothing=args.init_smoothing,
        output_2nd=args.output_2nd,
    )
