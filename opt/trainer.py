"""
The full training loop, now encapsulated in a class.
"""
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import psutil
import tensorflow as tf

from data_loading import (build_reaction_list, make_feature_matrices,
                          read_grid_points_for_molecule)
from auxiliary_functions import (save_weights_turbomole, extract_mols)

Reaction = Tuple[str, str, str, tf.Tensor, Dict]


class Trainer:

    KCAL = tf.constant(627.5096080305927, dtype=tf.float32)

    def __init__(self, model, optimizer, testset_objects, s6, s8, a1, a2, testset_names, testset_paths, scale, features, neg_punish, epochs):
        self.model = model
        self.optimizer = optimizer
        self.reactions: List[Reaction] = build_reaction_list(testset_objects, a1, a2, testset_names)
        self.s6 = s6
        self.s8 = s8
        self.epochs = epochs
        self.testset_paths = testset_paths
        self.scale = scale
        self.testset_names = testset_names
        self.features = features
        self.neg_punish = neg_punish
        self.group_size = {name: sum(1 for ts, *_ in self.reactions if ts == name)
                           for name in testset_names}
        self.process = psutil.Process(os.getpid())

    # ──────────────────────────────────────────────────────────

    def _reaction_forward(self, batch: Reaction) -> tf.Tensor:
        """
        Forward loss for one reaction; returns scalar loss (no grad).
        """
        ts_name, _r_id, expr, exp_kcal, d4 = batch
        mols = extract_mols(expr)

        x_train_full = np.empty((0, 28))
        w_lst, e1e_lst, d4_lst = [], [], []
        for mol in mols:
            e1e, feat, w, *_ = read_grid_points_for_molecule(ts_name, mol,self.testset_paths, False)
            x_train_full = np.concatenate((x_train_full, feat.T))
            w_lst.append(tf.constant(w, dtype=tf.float32))
            e1e_lst.append(tf.constant(e1e, dtype=tf.float32))
            d4_lst.append(
                tf.constant(self.s6 * d4[mol]['s6']
                            + self.s8 * d4[mol]['s8']
                            + d4[mol]['s9'],
                            dtype=tf.float32)
            )

        xm, xf = make_feature_matrices(x_train_full, self.features)
        xc_block, lmf_ = self.model(
            [tf.convert_to_tensor(xm), tf.convert_to_tensor(xf)],
            training=True
        )
        exc1, exc2 = tf.split(xc_block[:, 0:1], 2, axis=0)
        y_pred = (exc1 + exc2) / 2.0

        # molecule energies
        offset = 0
        e_mol_pred = {}
        for idx, mol in enumerate(mols):
            npts = w_lst[idx].shape[0]
            exc_slice = tf.squeeze(y_pred[offset: offset + npts], axis=-1)
            offset += npts
            e_mol_pred[mol] = e1e_lst[idx] + tf.reduce_sum(exc_slice * w_lst[idx]) + d4_lst[idx]

        # reaction energy
        expr_tf = expr
        for m in mols:
            expr_tf = expr_tf.replace(f"['{m}']", f"e_mol_pred['{m}']")
        e_react_pred = eval(expr_tf)
        pred_kcal = e_react_pred * self.KCAL

        loss = tf.abs(pred_kcal - exp_kcal)
        return loss, lmf_, ts_name

    # ──────────────────────────────────────────────────────────

    def train(self, checkpoint_dir: Path = Path(".")):
        """
        Main epochs batch loop.
        """
        for epoch in range(1, self.epochs + 1):
            random.shuffle(self.reactions)

            loss_sum = defaultdict(float)

            for batch in self.reactions:
                with tf.GradientTape() as tape:
                    loss, lmf_, ts_name = self._reaction_forward(batch)
                    scaled = loss * self.scale[ts_name] / self.group_size[ts_name] / 100

                    lmf_penalty_zero = self.neg_punish * tf.reduce_mean(tf.square(tf.nn.relu(-lmf_)))
                    if ts_name != 'FracP':
                        scaled += lmf_penalty_zero

                grads = tape.gradient(scaled, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                # bookkeeping for WTMAD-2
                loss_sum[ts_name] += float(loss)

            self._log_epoch(epoch, loss_sum)

            if epoch % 10 == 0 or epoch == self.epochs:
                self._save_checkpoint(epoch, checkpoint_dir)

        # final TurboMole dump
        save_weights_turbomole(self.model, output_path=checkpoint_dir / "final_weights")

    # ──────────────────────────────────────────────────────────
    # helpers
    # ──────────────────────────────────────────────────────────

    def _log_epoch(self, epoch: int, loss_sum: Dict[str, float]):
        log_parts = []
        wtmad_2 = 0.0
        for name in self.testset_names:
            avg_mae = loss_sum[name] / self.group_size[name]
            if name not in ('W417', 'BH76_full', 'FracP'):
                wtmad_2 += avg_mae * self.scale[name] / 100
            log_parts.append(f"{name}: {avg_mae:7.3f} kcal/mol")

        ram = self.process.memory_info().rss / 1024**2
        print(f"Epoch {epoch:3d}/{self.epochs} | RAM {ram:7.1f} MB | "
              + " | ".join(log_parts))
        print(f"WTMAD-2 {wtmad_2:7.1f} kcal/mol")

    def _save_checkpoint(self, epoch: int, checkpoint_dir: Path):
        tag = checkpoint_dir / f"weights_{epoch:05d}"
        save_weights_turbomole(self.model, output_path=f"{tag}.tm")
        self.model.save_weights(f"{tag}.weights.h5")
        print(f" ↳ saved {tag}.tm & {tag}.h5")
