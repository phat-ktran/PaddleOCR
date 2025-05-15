# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from rapidfuzz.distance import Levenshtein
from difflib import SequenceMatcher

import numpy as np
import string
import paddle
from .bleu import compute_bleu_score, compute_edit_distance


class RecMetric(object):
    def __init__(
        self, main_indicator="acc", is_filter=False, ignore_space=True, **kwargs
    ):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.eps = 1e-5
        self.reset()

    def _normalize_text(self, text):
        text = "".join(
            filter(lambda x: x in (string.digits + string.ascii_letters), text)
        )
        return text.lower()

    def __call__(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        correct_num, correct_char_num = 0, 0
        all_num, all_char_num = 0, 0
        norm_edit_dis = 0.0
        for (pred, pred_conf), (target, _) in zip(preds, labels):
            if self.ignore_space:
                pred = pred.replace(" ", "")
                target = target.replace(" ", "")
            if self.is_filter:
                pred = self._normalize_text(pred)
                target = self._normalize_text(target)
            norm_edit_dis += Levenshtein.normalized_distance(pred, target)
            if pred == target:
                correct_num += 1
            max_len = max(len(target), len(pred))
            for i in range(max_len):
                pred_c = pred[i] if len(pred) > i else None
                target_c = target[i] if len(target) > i else None
                if pred_c == target_c:
                    correct_char_num += 1
            all_char_num += len(target)
            all_num += 1
        self.correct_num += correct_num
        self.correct_char_num += correct_char_num
        self.all_num += all_num
        self.all_char_num += all_char_num
        self.norm_edit_dis += norm_edit_dis
        return {
            "acc": correct_num / (all_num + self.eps),
            "char_acc": correct_char_num / (all_char_num + self.eps),
            "norm_edit_dis": 1 - norm_edit_dis / (all_num + self.eps),
        }

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        char_acc = 1.0 * self.correct_char_num / (self.all_char_num + self.eps)
        norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + self.eps)
        self.reset()
        return {"acc": acc, "char_acc": char_acc, "norm_edit_dis": norm_edit_dis}

    def reset(self):
        self.correct_num = 0
        self.correct_char_num = 0
        self.all_num = 0
        self.all_char_num = 0
        self.norm_edit_dis = 0


class DistributedRecMetric(object):
    def __init__(
        self, main_indicator="acc", is_filter=False, ignore_space=True, **kwargs
    ):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.eps = 1e-5
        self.reset()

    def _normalize_text(self, text):
        text = "".join(
            filter(lambda x: x in (string.digits + string.ascii_letters), text)
        )
        return text.lower()

    def __call__(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        correct_num, correct_char_num = 0, 0
        all_num, all_char_num = 0, 0
        norm_edit_dis = 0.0
        for (pred, pred_conf), (target, _) in zip(preds, labels):
            if self.ignore_space:
                pred = pred.replace(" ", "")
                target = target.replace(" ", "")
            if self.is_filter:
                pred = self._normalize_text(pred)
                target = self._normalize_text(target)
            norm_edit_dis += Levenshtein.normalized_distance(pred, target)
            if pred == target:
                correct_num += 1
            max_len = max(len(target), len(pred))
            for i in range(max_len):
                pred_c = pred[i] if len(pred) > i else None
                target_c = target[i] if len(target) > i else None
                if pred_c == target_c:
                    correct_char_num += 1
            all_char_num += len(target)
            all_num += 1
        self.correct_num += correct_num
        self.correct_char_num += correct_char_num
        self.all_num += all_num
        self.all_char_num += all_char_num
        self.norm_edit_dis += norm_edit_dis

    def gather_metrics(self):
        # Gather metrics across all processes
        metrics = {
            "correct_num": paddle.to_tensor(self.correct_num),
            "correct_char_num": paddle.to_tensor(self.correct_char_num),
            "all_num": paddle.to_tensor(self.all_num),
            "all_char_num": paddle.to_tensor(self.all_char_num),
            "norm_edit_dis": paddle.to_tensor(self.norm_edit_dis),
        }
        if paddle.distributed.get_world_size() > 1:  # Check if distributed mode is enabled
            for key, tensor in metrics.items():
                paddle.distributed.all_reduce(tensor, op=paddle.distributed.ReduceOp.SUM)
                metrics[key] = tensor.numpy()[0]
        else:
            metrics = {key: tensor.numpy()[0] for key, tensor in metrics.items()}
        return metrics

    def get_metric(self):
        # Aggregate metrics
        metrics = self.gather_metrics()
        acc = metrics["correct_num"] / (metrics["all_num"] + self.eps)
        char_acc = metrics["correct_char_num"] / (metrics["all_char_num"] + self.eps)
        norm_edit_dis = 1 - metrics["norm_edit_dis"] / (metrics["all_num"] + self.eps)
        self.reset()
        return {"acc": acc, "char_acc": char_acc, "norm_edit_dis": norm_edit_dis}

    def reset(self):
        self.correct_num = 0
        self.correct_char_num = 0
        self.all_num = 0
        self.all_char_num = 0
        self.norm_edit_dis = 0


class CNTMetric(object):
    def __init__(self, main_indicator="acc", **kwargs):
        self.main_indicator = main_indicator
        self.eps = 1e-5
        self.reset()

    def __call__(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        correct_num = 0
        all_num = 0
        for pred, target in zip(preds, labels):
            if pred == target:
                correct_num += 1
            all_num += 1
        self.correct_num += correct_num
        self.all_num += all_num
        return {
            "acc": correct_num / (all_num + self.eps),
        }

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
            }
        """
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        self.reset()
        return {"acc": acc}

    def reset(self):
        self.correct_num = 0
        self.all_num = 0


class CANMetric(object):
    def __init__(self, main_indicator="exp_rate", **kwargs):
        self.main_indicator = main_indicator
        self.word_right = []
        self.exp_right = []
        self.word_total_length = 0
        self.exp_total_num = 0
        self.word_rate = 0
        self.exp_rate = 0
        self.reset()
        self.epoch_reset()

    def __call__(self, preds, batch, **kwargs):
        for k, v in kwargs.items():
            epoch_reset = v
            if epoch_reset:
                self.epoch_reset()
        word_probs = preds
        word_label, word_label_mask = batch
        line_right = 0
        if word_probs is not None:
            word_pred = word_probs.argmax(2)
        word_pred = word_pred.cpu().detach().numpy()
        word_scores = [
            SequenceMatcher(
                None, s1[: int(np.sum(s3))], s2[: int(np.sum(s3))], autojunk=False
            ).ratio()
            * (len(s1[: int(np.sum(s3))]) + len(s2[: int(np.sum(s3))]))
            / len(s1[: int(np.sum(s3))])
            / 2
            for s1, s2, s3 in zip(word_label, word_pred, word_label_mask)
        ]
        batch_size = len(word_scores)
        for i in range(batch_size):
            if word_scores[i] == 1:
                line_right += 1
        self.word_rate = np.mean(word_scores)  # float
        self.exp_rate = line_right / batch_size  # float
        exp_length, word_length = word_label.shape[:2]
        self.word_right.append(self.word_rate * word_length)
        self.exp_right.append(self.exp_rate * exp_length)
        self.word_total_length = self.word_total_length + word_length
        self.exp_total_num = self.exp_total_num + exp_length

    def get_metric(self):
        """
        return {
            'word_rate': 0,
            "exp_rate": 0,
        }
        """
        cur_word_rate = sum(self.word_right) / self.word_total_length
        cur_exp_rate = sum(self.exp_right) / self.exp_total_num
        self.reset()
        return {"word_rate": cur_word_rate, "exp_rate": cur_exp_rate}

    def reset(self):
        self.word_rate = 0
        self.exp_rate = 0

    def epoch_reset(self):
        self.word_right = []
        self.exp_right = []
        self.word_total_length = 0
        self.exp_total_num = 0


class LaTeXOCRMetric(object):
    def __init__(self, main_indicator="exp_rate", cal_bleu_score=False, **kwargs):
        self.main_indicator = main_indicator
        self.cal_bleu_score = cal_bleu_score
        self.edit_right = []
        self.exp_right = []
        self.bleu_right = []
        self.e1_right = []
        self.e2_right = []
        self.e3_right = []
        self.editdistance_total_length = 0
        self.exp_total_num = 0
        self.edit_dist = 0
        self.exp_rate = 0
        if self.cal_bleu_score:
            self.bleu_score = 0
        self.e1 = 0
        self.e2 = 0
        self.e3 = 0
        self.reset()
        self.epoch_reset()

    def __call__(self, preds, batch, **kwargs):
        for k, v in kwargs.items():
            epoch_reset = v
            if epoch_reset:
                self.epoch_reset()
        word_pred = preds
        word_label = batch
        line_right, e1, e2, e3 = 0, 0, 0, 0
        lev_dist = []
        for labels, prediction in zip(word_label, word_pred):
            if prediction == labels:
                line_right += 1
            distance = compute_edit_distance(prediction, labels)
            lev_dist.append(Levenshtein.normalized_distance(prediction, labels))
            if distance <= 1:
                e1 += 1
            if distance <= 2:
                e2 += 1
            if distance <= 3:
                e3 += 1

        batch_size = len(lev_dist)

        self.edit_dist = sum(lev_dist)  # float
        self.exp_rate = line_right  # float
        if self.cal_bleu_score:
            self.bleu_score = compute_bleu_score(word_pred, word_label)
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3
        exp_length = len(word_label)
        self.edit_right.append(self.edit_dist)
        self.exp_right.append(self.exp_rate)
        if self.cal_bleu_score:
            self.bleu_right.append(self.bleu_score * batch_size)
        self.e1_right.append(self.e1)
        self.e2_right.append(self.e2)
        self.e3_right.append(self.e3)
        self.editdistance_total_length = self.editdistance_total_length + exp_length
        self.exp_total_num = self.exp_total_num + exp_length

    def get_metric(self):
        """
        return {
            'edit distance': 0,
            "bleu_score": 0,
            "exp_rate": 0,
        }
        """
        cur_edit_distance = sum(self.edit_right) / self.exp_total_num
        cur_exp_rate = sum(self.exp_right) / self.exp_total_num
        if self.cal_bleu_score:
            cur_bleu_score = sum(self.bleu_right) / self.editdistance_total_length
        cur_exp_1 = sum(self.e1_right) / self.exp_total_num
        cur_exp_2 = sum(self.e2_right) / self.exp_total_num
        cur_exp_3 = sum(self.e3_right) / self.exp_total_num
        self.reset()
        if self.cal_bleu_score:
            return {
                "bleu_score": cur_bleu_score,
                "edit distance": cur_edit_distance,
                "exp_rate": cur_exp_rate,
                "exp_rate<=1 ": cur_exp_1,
                "exp_rate<=2 ": cur_exp_2,
                "exp_rate<=3 ": cur_exp_3,
            }
        else:
            return {
                "edit distance": cur_edit_distance,
                "exp_rate": cur_exp_rate,
                "exp_rate<=1 ": cur_exp_1,
                "exp_rate<=2 ": cur_exp_2,
                "exp_rate<=3 ": cur_exp_3,
            }

    def reset(self):
        self.edit_dist = 0
        self.exp_rate = 0
        if self.cal_bleu_score:
            self.bleu_score = 0
        self.e1 = 0
        self.e2 = 0
        self.e3 = 0

    def epoch_reset(self):
        self.edit_right = []
        self.exp_right = []
        if self.cal_bleu_score:
            self.bleu_right = []
        self.e1_right = []
        self.e2_right = []
        self.e3_right = []
        self.editdistance_total_length = 0
        self.exp_total_num = 0
