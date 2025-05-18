# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle

__all__ = [
    "DetMetric",
    "DistributedDetMetric",
    "DetFCEMetric",
    "DistributedDetFCEMetric",
]

from .eval_det_iou import DetectionIoUEvaluator


class DetMetric(object):
    def __init__(self, main_indicator="hmean", **kwargs):
        self.evaluator = DetectionIoUEvaluator()
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch, **kwargs):
        """
        batch: a list produced by dataloaders.
            image: np.ndarray  of shape (N, C, H, W).
            ratio_list: np.ndarray  of shape(N,2)
            polygons: np.ndarray  of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: np.ndarray  of shape (N, K), indicates whether a region is ignorable or not.
        preds: a list of dict produced by post process
             points: np.ndarray of shape (N, K, 4, 2), the polygons of objective regions.
        """
        gt_polyons_batch = batch[2]
        ignore_tags_batch = batch[3]
        for pred, gt_polyons, ignore_tags in zip(
            preds, gt_polyons_batch, ignore_tags_batch
        ):
            # prepare gt
            gt_info_list = [
                {"points": gt_polyon, "text": "", "ignore": ignore_tag}
                for gt_polyon, ignore_tag in zip(gt_polyons, ignore_tags)
            ]
            # prepare det
            det_info_list = [
                {"points": det_polyon, "text": ""} for det_polyon in pred["points"]
            ]
            result = self.evaluator.evaluate_image(gt_info_list, det_info_list)
            self.results.append(result)

    def get_metric(self):
        """
        return metrics {
                 'precision': 0,
                 'recall': 0,
                 'hmean': 0
            }
        """

        metrics = self.evaluator.combine_results(self.results)
        self.reset()
        return metrics

    def reset(self):
        self.results = []  # clear results


class DistributedDetMetric(object):
    def __init__(self, main_indicator="hmean", **kwargs):
        self.evaluator = DetectionIoUEvaluator()
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch, **kwargs):
        """
        batch: a list produced by dataloaders.
            image: np.ndarray  of shape (N, C, H, W).
            ratio_list: np.ndarray  of shape(N,2)
            polygons: np.ndarray  of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: np.ndarray  of shape (N, K), indicates whether a region is ignorable or not.
        preds: a list of dict produced by post process
             points: np.ndarray of shape (N, K, 4, 2), the polygons of objective regions.
        """
        gt_polyons_batch = batch[2]
        ignore_tags_batch = batch[3]
        for pred, gt_polyons, ignore_tags in zip(
            preds, gt_polyons_batch, ignore_tags_batch
        ):
            # prepare gt
            gt_info_list = [
                {"points": gt_polyon, "text": "", "ignore": ignore_tag}
                for gt_polyon, ignore_tag in zip(gt_polyons, ignore_tags)
            ]
            # prepare det
            det_info_list = [
                {"points": det_polyon, "text": ""} for det_polyon in pred["points"]
            ]
            result = self.evaluator.evaluate_image(gt_info_list, det_info_list)
            self.results.append(result)

    def gather_metrics(self):
        """
        Gather results from all distributed processes and combine them.
        """
        all_results = []
        paddle.distributed.all_gather_object(all_results, self.results)

        # all_results is now a list of lists; flatten it:
        flat_results = [item for sublist in all_results for item in sublist]

        # 2) Combine exactly once:
        metrics = self.evaluator.combine_results(flat_results)
        return metrics

    def get_metric(self):
        """
        return metrics {
                 'precision': 0,
                 'recall': 0,
                 'hmean': 0
            }
        """
        # If we're in distributed mode, we need to gather results first
        metrics = self.gather_metrics()

        self.reset()

        return metrics

    def reset(self):
        self.results = []  # clear results


class DetFCEMetric(object):
    def __init__(self, main_indicator="hmean", **kwargs):
        self.evaluator = DetectionIoUEvaluator()
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch, **kwargs):
        """
        batch: a list produced by dataloaders.
            image: np.ndarray  of shape (N, C, H, W).
            ratio_list: np.ndarray  of shape(N,2)
            polygons: np.ndarray  of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: np.ndarray  of shape (N, K), indicates whether a region is ignorable or not.
        preds: a list of dict produced by post process
             points: np.ndarray of shape (N, K, 4, 2), the polygons of objective regions.
        """
        gt_polyons_batch = batch[2]
        ignore_tags_batch = batch[3]

        for pred, gt_polyons, ignore_tags in zip(
            preds, gt_polyons_batch, ignore_tags_batch
        ):
            # prepare gt
            gt_info_list = [
                {"points": gt_polyon, "text": "", "ignore": ignore_tag}
                for gt_polyon, ignore_tag in zip(gt_polyons, ignore_tags)
            ]
            # prepare det
            det_info_list = [
                {"points": det_polyon, "text": "", "score": score}
                for det_polyon, score in zip(pred["points"], pred["scores"])
            ]

            for score_thr in self.results.keys():
                det_info_list_thr = [
                    det_info
                    for det_info in det_info_list
                    if det_info["score"] >= score_thr
                ]
                result = self.evaluator.evaluate_image(gt_info_list, det_info_list_thr)
                self.results[score_thr].append(result)

    def get_metric(self):
        """
        return metrics {'heman':0,
            'thr 0.3':'precision: 0 recall: 0 hmean: 0',
            'thr 0.4':'precision: 0 recall: 0 hmean: 0',
            'thr 0.5':'precision: 0 recall: 0 hmean: 0',
            'thr 0.6':'precision: 0 recall: 0 hmean: 0',
            'thr 0.7':'precision: 0 recall: 0 hmean: 0',
            'thr 0.8':'precision: 0 recall: 0 hmean: 0',
            'thr 0.9':'precision: 0 recall: 0 hmean: 0',
            }
        """
        metrics = {}
        hmean = 0
        for score_thr in self.results.keys():
            metric = self.evaluator.combine_results(self.results[score_thr])
            # for key, value in metric.items():
            #     metrics['{}_{}'.format(key, score_thr)] = value
            metric_str = "precision:{:.5f} recall:{:.5f} hmean:{:.5f}".format(
                metric["precision"], metric["recall"], metric["hmean"]
            )
            metrics["thr {}".format(score_thr)] = metric_str
            hmean = max(hmean, metric["hmean"])
        metrics["hmean"] = hmean

        self.reset()
        return metrics

    def reset(self):
        self.results = {
            0.3: [],
            0.4: [],
            0.5: [],
            0.6: [],
            0.7: [],
            0.8: [],
            0.9: [],
        }  # clear results


class DistributedDetFCEMetric(object):
    def __init__(self, main_indicator="hmean", **kwargs):
        self.evaluator = DetectionIoUEvaluator()
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch, **kwargs):
        """
        batch: a list produced by dataloaders.
            image: np.ndarray  of shape (N, C, H, W).
            ratio_list: np.ndarray  of shape(N,2)
            polygons: np.ndarray  of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: np.ndarray  of shape (N, K), indicates whether a region is ignorable or not.
        preds: a list of dict produced by post process
             points: np.ndarray of shape (N, K, 4, 2), the polygons of objective regions.
        """
        gt_polyons_batch = batch[2]
        ignore_tags_batch = batch[3]

        for pred, gt_polyons, ignore_tags in zip(
            preds, gt_polyons_batch, ignore_tags_batch
        ):
            # prepare gt
            gt_info_list = [
                {"points": gt_polyon, "text": "", "ignore": ignore_tag}
                for gt_polyon, ignore_tag in zip(gt_polyons, ignore_tags)
            ]
            # prepare det
            det_info_list = [
                {"points": det_polyon, "text": "", "score": score}
                for det_polyon, score in zip(pred["points"], pred["scores"])
            ]

            for score_thr in self.results.keys():
                det_info_list_thr = [
                    det_info
                    for det_info in det_info_list
                    if det_info["score"] >= score_thr
                ]
                result = self.evaluator.evaluate_image(gt_info_list, det_info_list_thr)
                self.results[score_thr].append(result)

    def gather_metrics(self):
        """
        Gather results from all distributed processes for each threshold
        """
        # Initialize metrics dictionary to store aggregated results for each threshold
        metrics_by_threshold = {}

        for score_thr in self.results.keys():
            # Count metrics for this threshold
            true_positives = paddle.to_tensor(
                sum(
                    r["true_positive_num"]
                    for r in self.results[score_thr]
                    if "true_positive_num" in r
                ),
                dtype="int64",
            )
            false_positives = paddle.to_tensor(
                sum(
                    r["false_positive_num"]
                    for r in self.results[score_thr]
                    if "false_positive_num" in r
                ),
                dtype="int64",
            )
            false_negatives = paddle.to_tensor(
                sum(
                    r["false_negative_num"]
                    for r in self.results[score_thr]
                    if "false_negative_num" in r
                ),
                dtype="int64",
            )

            metrics = {
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
            }

            # Perform all_reduce operation if in distributed mode
            if paddle.distributed.get_world_size() > 1:
                for key, tensor in metrics.items():
                    paddle.distributed.all_reduce(
                        tensor, op=paddle.distributed.ReduceOp.SUM
                    )
                    metrics[key] = tensor.numpy().item()  # Extract scalar value
            else:
                metrics = {
                    key: tensor.numpy().item() for key, tensor in metrics.items()
                }

            metrics_by_threshold[score_thr] = metrics

        return metrics_by_threshold

    def get_metric(self):
        """
        return metrics {'hmean':0,
            'thr 0.3':'precision: 0 recall: 0 hmean: 0',
            'thr 0.4':'precision: 0 recall: 0 hmean: 0',
            'thr 0.5':'precision: 0 recall: 0 hmean: 0',
            'thr 0.6':'precision: 0 recall: 0 hmean: 0',
            'thr 0.7':'precision: 0 recall: 0 hmean: 0',
            'thr 0.8':'precision: 0 recall: 0 hmean: 0',
            'thr 0.9':'precision: 0 recall: 0 hmean: 0',
            }
        """
        # Gather metrics across all distributed processes
        metrics_by_threshold = self.gather_metrics()

        # Calculate final metrics
        metrics = {}
        hmean = 0

        for score_thr, agg_metrics in metrics_by_threshold.items():
            tp = agg_metrics["true_positives"]
            fp = agg_metrics["false_positives"]
            fn = agg_metrics["false_negatives"]

            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            current_hmean = 0.0
            if precision + recall > 0:
                current_hmean = 2.0 * precision * recall / (precision + recall)

            metric_str = "precision:{:.5f} recall:{:.5f} hmean:{:.5f}".format(
                precision, recall, current_hmean
            )
            metrics["thr {}".format(score_thr)] = metric_str
            hmean = max(hmean, current_hmean)

        metrics["hmean"] = hmean
        self.reset()
        return metrics

    def reset(self):
        self.results = {
            0.3: [],
            0.4: [],
            0.5: [],
            0.6: [],
            0.7: [],
            0.8: [],
            0.9: [],
        }  # clear results
