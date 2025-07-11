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

from collections import defaultdict
import os
import numpy as np
import paddle
from paddle.nn import functional as F
import re
import json


class BaseRecLabelDecode(object):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, use_space_char=False):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str = []

        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
            if "arabic" in character_dict_path:
                self.reverse = True

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def pred_reverse(self, pred):
        pred_re = []
        c_current = ""
        for c in pred:
            if not bool(re.search("[a-zA-Z0-9 :*./%+-]", c)):
                if c_current != "":
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ""
            else:
                c_current += c
        if c_current != "":
            pred_re.append(c_current)

        return "".join(pred_re[::-1])

    def add_special_char(self, dict_character):
        return dict_character

    def get_word_info(self, text, selection):
        """
        Group the decoded characters and record the corresponding decoded positions.

        Args:
            text: the decoded text
            selection: the bool array that identifies which columns of features are decoded as non-separated characters
        Returns:
            word_list: list of the grouped words
            word_col_list: list of decoding positions corresponding to each character in the grouped word
            state_list: list of marker to identify the type of grouping words, including two types of grouping words:
                        - 'cn': continuous chinese characters (e.g., 你好啊)
                        - 'en&num': continuous english characters (e.g., hello), number (e.g., 123, 1.123), or mixed of them connected by '-' (e.g., VGG-16)
                        The remaining characters in text are treated as separators between groups (e.g., space, '(', ')', etc.).
        """
        state = None
        word_content = []
        word_col_content = []
        word_list = []
        word_col_list = []
        state_list = []
        valid_col = np.where(selection is True)[0]

        for c_i, char in enumerate(text):
            if "\u4e00" <= char <= "\u9fff":
                c_state = "cn"
            elif bool(re.search("[a-zA-Z0-9]", char)):
                c_state = "en&num"
            else:
                c_state = "splitter"

            if (
                char == "."
                and state == "en&num"
                and c_i + 1 < len(text)
                and bool(re.search("[0-9]", text[c_i + 1]))
            ):  # grouping floating number
                c_state = "en&num"
            if (
                char == "-" and state == "en&num"
            ):  # grouping word with '-', such as 'state-of-the-art'
                c_state = "en&num"

            if state is None:
                state = c_state

            if state != c_state:
                if len(word_content) != 0:
                    word_list.append(word_content)
                    word_col_list.append(word_col_content)
                    state_list.append(state)
                    word_content = []
                    word_col_content = []
                state = c_state

            if state != "splitter":
                word_content.append(char)
                word_col_content.append(valid_col[c_i])

        if len(word_content) != 0:
            word_list.append(word_content)
            word_col_list.append(word_col_content)
            state_list.append(state)

        return word_list, word_col_list, state_list

    def decode(
        self,
        text_index,
        text_prob=None,
        is_remove_duplicate=False,
        return_word_box=False,
    ):
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id] for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = "".join(char_list)

            if self.reverse:  # for arabic rec
                text = self.pred_reverse(text)

            if return_word_box:
                word_list, word_col_list, state_list = self.get_word_info(
                    text, selection
                )
                result_list.append(
                    (
                        text,
                        np.mean(conf_list).tolist(),
                        [
                            len(text_index[batch_idx]),
                            word_list,
                            word_col_list,
                            state_list,
                        ],
                    )
                )
            else:
                result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path, use_space_char)
        self.is_remove_duplicate = kwargs.get("is_remove_duplicate", True)

    def __call__(self, preds, label=None, return_word_box=False, *args, **kwargs):
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(
            preds_idx,
            preds_prob,
            is_remove_duplicate=self.is_remove_duplicate,
            return_word_box=return_word_box,
        )
        if return_word_box:
            for rec_idx, rec in enumerate(text):
                wh_ratio = kwargs["wh_ratio_list"][rec_idx]
                max_wh_ratio = kwargs["max_wh_ratio"]
                rec[2][0] = rec[2][0] * (wh_ratio / max_wh_ratio)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ["blank"] + dict_character
        return dict_character


class BeamCTCLabelDecode(BaseRecLabelDecode):
    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super(BeamCTCLabelDecode, self).__init__(character_dict_path, use_space_char)

    def get_topk_characters(self, probs, k=5):
        """
        Get top-k characters and their probabilities for a given timestep

        Args:
            probs: probability array for a single timestep [num_classes]
            k: number of top characters to return

        Returns:
            List of tuples (char_idx, char_symbol, probability) sorted by probability desc
        """
        # Get top-k indices and their probabilities
        topk_indices = np.argsort(probs)[-k:][::-1]  # Sort descending
        topk_chars = []

        for idx in topk_indices:
            prob = probs[idx]
            if idx == 0:
                char_symbol = "<blank>"
            elif idx < len(self.character):
                char_symbol = self.character[idx]
            else:
                char_symbol = f"<unk_{idx}>"

            topk_chars.append((char_symbol, float(prob)))

        return topk_chars

    def beam_search_decode(self, preds, beam_width=5, return_all_beams=False, topk_chars=5):
        """
        CTC beam search decoder with detailed timestep tracking and top-k character collection

        Args:
            preds: prediction probabilities of shape [batch_size, seq_len, num_classes]
            beam_width: number of beams to keep during search
            return_all_beams: if True, return all beams; if False, return only the best beam
            topk_chars: number of top characters to collect for each alignment timestep

        Returns:
            If return_all_beams=False: List of tuples (decoded_indices, decoded_probabilities, timestep_info) for each batch
            If return_all_beams=True: List of lists, where each inner list contains all beam candidates
                                     as tuples (decoded_indices, decoded_probabilities, rank, timestep_info)
        """
        batch_size, seq_len, num_classes = preds.shape
        blank_id = 0  # CTC blank token

        results = []

        for batch_idx in range(batch_size):
            # Initialize beam with empty sequence
            # Each beam item: (log_prob, sequence, last_char, timestep_probs, alignment, topk_per_alignment)
            # timestep_probs: list of probabilities for each character in sequence
            # alignment: list of timestep indices where each character was added
            # topk_per_alignment: list of top-k characters for each alignment timestep
            beams = [(0.0, [], -1, [], [], [])]

            # Collect top-k characters for each timestep for reference
            timestep_topk_all = []
            for t in range(seq_len):
                timestep_probs = preds[batch_idx, t, :]
                topk_chars_t = self.get_topk_characters(timestep_probs, topk_chars)
                timestep_topk_all.append(topk_chars_t)

            for t in range(seq_len):
                new_beams = defaultdict(lambda: (float("-inf"), [], [], []))

                for log_prob, seq, last_char, timestep_probs, alignment, topk_per_alignment in beams:
                    for c in range(num_classes):
                        char_prob = preds[batch_idx, t, c]
                        char_log_prob = np.log(char_prob + 1e-8)
                        new_log_prob = log_prob + char_log_prob

                        if c == blank_id:
                            # Blank token - no character added
                            key = (tuple(seq), last_char)
                            if new_log_prob > new_beams[key][0]:
                                new_beams[key] = (
                                    new_log_prob,
                                    timestep_probs.copy(),
                                    alignment.copy(),
                                    topk_per_alignment.copy(),
                                )
                        else:
                            # Non-blank token
                            if c == last_char:
                                # Same character as last - only add if we had a blank or it's different
                                if len(seq) == 0 or seq[-1] != c:
                                    new_seq = seq + [c]
                                    new_timestep_probs = timestep_probs + [char_prob]
                                    new_alignment = alignment + [t]
                                    new_topk_per_alignment = topk_per_alignment + [timestep_topk_all[t]]
                                    key = (tuple(new_seq), c)
                                    if new_log_prob > new_beams[key][0]:
                                        new_beams[key] = (
                                            new_log_prob,
                                            new_timestep_probs,
                                            new_alignment,
                                            new_topk_per_alignment,
                                        )
                                else:
                                    # Repeat character, keep original sequence but update last character's probability if better
                                    key = (tuple(seq), c)
                                    updated_probs = timestep_probs.copy()
                                    updated_alignment = alignment.copy()
                                    updated_topk = topk_per_alignment.copy()

                                    # Update the last character's info if this timestep is more confident
                                    if (
                                        len(updated_probs) > 0
                                        and char_prob > updated_probs[-1]
                                    ):
                                        updated_probs[-1] = char_prob
                                        updated_alignment[-1] = t
                                        updated_topk[-1] = timestep_topk_all[t]

                                    if new_log_prob > new_beams[key][0]:
                                        new_beams[key] = (
                                            new_log_prob,
                                            updated_probs,
                                            updated_alignment,
                                            updated_topk,
                                        )
                            else:
                                # Different character - add to sequence
                                new_seq = seq + [c]
                                new_timestep_probs = timestep_probs + [char_prob]
                                new_alignment = alignment + [t]
                                new_topk_per_alignment = topk_per_alignment + [timestep_topk_all[t]]
                                key = (tuple(new_seq), c)
                                if new_log_prob > new_beams[key][0]:
                                    new_beams[key] = (
                                        new_log_prob,
                                        new_timestep_probs,
                                        new_alignment,
                                        new_topk_per_alignment,
                                    )

                # Keep top beam_width beams
                beams = []
                for (seq, last_char), (log_prob, timestep_probs, alignment, topk_per_alignment) in sorted(
                    new_beams.items(), key=lambda x: x[1][0], reverse=True
                )[:beam_width]:
                    beams.append(
                        (log_prob, list(seq), last_char, timestep_probs, alignment, topk_per_alignment)
                    )

            # Process results based on return_all_beams flag
            if return_all_beams:
                # Return all beams with their rankings and timestep info
                beam_candidates = []
                if beams:
                    # Sort beams by probability (highest first)
                    sorted_beams = sorted(beams, key=lambda x: x[0], reverse=True)
                    for rank, (
                        log_prob,
                        seq,
                        _,
                        timestep_probs,
                        alignment,
                        topk_per_alignment,
                    ) in enumerate(sorted_beams):
                        prob = np.exp(log_prob)
                        timestep_info = {
                            "alignment": alignment,
                            "topk_per_alignment": topk_per_alignment,
                            "avg_char_conf": np.mean(timestep_probs)
                            if timestep_probs
                            else 0.0,
                            "min_char_conf": np.min(timestep_probs)
                            if timestep_probs
                            else 0.0,
                            "max_char_conf": np.max(timestep_probs)
                            if timestep_probs
                            else 0.0,
                            "timestep_topk_all": timestep_topk_all,  # All timesteps top-k for reference
                        }
                        beam_candidates.append((seq, prob, rank + 1, timestep_info))
                else:
                    timestep_info = {
                        "alignment": [],
                        "topk_per_alignment": [],
                        "avg_char_conf": 0.0,
                        "min_char_conf": 0.0,
                        "max_char_conf": 0.0,
                        "timestep_topk_all": timestep_topk_all,
                    }
                    beam_candidates.append(([], 0.0, 1, timestep_info))
                results.append(beam_candidates)
            else:
                # Return only the best beam (original behavior)
                if beams:
                    best_log_prob, best_seq, _, best_timestep_probs, best_alignment, best_topk_per_alignment = (
                        max(beams, key=lambda x: x[0])
                    )
                    best_prob = np.exp(best_log_prob)
                    timestep_info = {
                        "alignment": best_alignment,
                        "topk_per_alignment": best_topk_per_alignment,
                        "avg_char_conf": np.mean(best_timestep_probs)
                        if best_timestep_probs
                        else 0.0,
                        "timestep_topk_all": timestep_topk_all,
                    }
                    results.append((best_seq, best_prob, timestep_info))
                else:
                    timestep_info = {
                        "alignment": [],
                        "topk_per_alignment": [],
                        "avg_char_conf": 0.0,
                        "timestep_topk_all": timestep_topk_all,
                    }
                    results.append(([], 0.0, timestep_info))

        return results

    def greedy_decode_with_alignment(self, preds, topk_chars=5):
        """
        Greedy decoding with alignment and top-k character tracking

        Args:
            preds: prediction probabilities of shape [batch_size, seq_len, num_classes]
            topk_chars: number of top characters to collect for each alignment timestep

        Returns:
            List of tuples (decoded_indices, decoded_probabilities, timestep_info) for each batch
        """
        batch_size, seq_len, num_classes = preds.shape
        blank_id = 0

        results = []

        for batch_idx in range(batch_size):
            # Greedy decoding
            pred_indices = preds[batch_idx].argmax(axis=1)  # [seq_len]
            pred_probs = preds[batch_idx].max(axis=1)  # [seq_len]

            # Track alignment and collect top-k
            alignment_timesteps = []
            alignment_char_probs = []
            topk_per_alignment = []
            decoded_sequence = []

            # Collect top-k for all timesteps
            timestep_topk_all = []
            for t in range(seq_len):
                timestep_probs = preds[batch_idx, t, :]
                topk_chars_t = self.get_topk_characters(timestep_probs, topk_chars)
                timestep_topk_all.append(topk_chars_t)

            # Process sequence to find alignments (similar to CTC collapse)
            prev_char = -1  # Previous non-blank character

            for t in range(seq_len):
                current_char = pred_indices[t]
                current_prob = pred_probs[t]

                if current_char != blank_id:  # Non-blank
                    if current_char != prev_char:  # New character (not a repeat)
                        # This timestep represents a new character alignment
                        decoded_sequence.append(current_char)
                        alignment_timesteps.append(t)
                        alignment_char_probs.append(current_prob)
                        topk_per_alignment.append(timestep_topk_all[t])

                    prev_char = current_char
                else:
                    # Blank token resets the previous character
                    prev_char = -1

            # Create timestep info
            timestep_info = {
                "alignment": alignment_timesteps,
                "topk_per_alignment": topk_per_alignment,
                "avg_char_conf": np.mean(alignment_char_probs) if alignment_char_probs else 0.0,
                "min_char_conf": np.min(alignment_char_probs) if alignment_char_probs else 0.0,
                "max_char_conf": np.max(alignment_char_probs) if alignment_char_probs else 0.0,
                "timestep_topk_all": timestep_topk_all,
            }

            # Calculate overall probability (geometric mean)
            if len(alignment_char_probs) > 0:
                overall_prob = np.prod(alignment_char_probs) ** (1.0 / len(alignment_char_probs))
            else:
                overall_prob = 1.0

            results.append((decoded_sequence, overall_prob, timestep_info))

        return results

    def __call__(
        self,
        preds,
        label=None,
        return_word_box=False,
        use_beam_search=False,
        beam_width=5,
        return_all_beams=False,
        topk_chars=5,
        return_alignment_info=False,
        *args,
        **kwargs,
    ):
        """
        Decode CTC predictions using either greedy or beam search

        Args:
            preds: prediction tensor/array
            label: ground truth labels (optional)
            return_word_box: whether to return word bounding box info
            use_beam_search: whether to use beam search instead of greedy decoding
            beam_width: beam width for beam search (only used if use_beam_search=True)
            return_all_beams: if True and use_beam_search=True, return all beam candidates
            topk_chars: number of top characters to collect for each alignment timestep
            return_alignment_info: if True, return alignment and top-k info even for greedy decoding

        Returns:
            If return_all_beams=False: Standard format (text, confidence) tuples, or with alignment info
            If return_all_beams=True: Dictionary with 'best_result' and 'all_beams' keys
            If return_alignment_info=True: Dictionary with alignment and top-k information
        """
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()

        if use_beam_search:
            # Use beam search decoding
            beam_results = self.beam_search_decode(preds, beam_width, return_all_beams, topk_chars)

            if return_all_beams:
                # Process all beams for each batch
                all_beams_results = []
                best_results = []

                for batch_beams in beam_results:
                    batch_all_beams = []
                    best_beam = None

                    for seq, prob, rank, timestep_info in batch_beams:
                        # Convert indices to text
                        if len(seq) == 0:
                            text = ""
                            conf = prob
                        else:
                            # Remove duplicates and blanks
                            filtered_seq = []
                            prev_char = -1
                            for char_idx in seq:
                                if (
                                    char_idx != 0 and char_idx != prev_char
                                ):  # 0 is blank
                                    filtered_seq.append(char_idx)
                                prev_char = char_idx

                            if len(filtered_seq) == 0:
                                text = ""
                                conf = prob
                            else:
                                char_list = [
                                    self.character[idx]
                                    for idx in filtered_seq
                                    if idx < len(self.character)
                                ]
                                text = "".join(char_list)

                                if self.reverse:  # for arabic rec
                                    text = self.pred_reverse(text)

                                conf = prob

                        beam_info = {
                            "text": text,
                            "confidence": conf,
                            "rank": rank,
                            "timestep_info": timestep_info
                        }

                        if return_word_box:
                            # Add word box info for each beam (simplified)
                            selection = (
                                np.ones(len(seq), dtype=bool)
                                if len(seq) > 0
                                else np.array([])
                            )
                            if len(text) > 0:
                                word_list, word_col_list, state_list = (
                                    self.get_word_info(text, selection)
                                )
                                beam_info["word_info"] = [
                                    len(seq),
                                    word_list,
                                    word_col_list,
                                    state_list,
                                ]

                        batch_all_beams.append(beam_info)

                        # Keep track of best beam (rank 1)
                        if rank == 1:
                            if return_word_box:
                                best_beam = (text, conf, beam_info.get("word_info", []))
                            else:
                                best_beam = (text, conf)

                    all_beams_results.append(batch_all_beams)
                    best_results.append(best_beam)

                # Return both all beams and best results
                result = {"best_result": best_results, "candidates": all_beams_results}

                if label is not None:
                    label_decoded = self.decode(label)
                    result["label"] = label_decoded

                return result

            else:
                # Return only best beam (original behavior)
                preds_idx = []
                preds_prob = []

                for seq, prob, timestep_info in beam_results:
                    if len(seq) == 0:
                        preds_idx.append([0])
                        preds_prob.append([1.0])
                    else:
                        preds_idx.append(seq)
                        avg_prob = prob ** (1.0 / len(seq)) if len(seq) > 0 else prob
                        preds_prob.append([avg_prob] * len(seq))

                # Pad sequences to same length
                max_len = max(len(seq) for seq in preds_idx) if preds_idx else 1
                for i in range(len(preds_idx)):
                    while len(preds_idx[i]) < max_len:
                        preds_idx[i].append(0)
                        preds_prob[i].append(0.0)

                preds_idx = np.array(preds_idx)
                preds_prob = np.array(preds_prob)
        else:
            # Use greedy decoding
            if return_alignment_info:
                # Use enhanced greedy decoding with alignment tracking
                greedy_results = self.greedy_decode_with_alignment(preds, topk_chars)

                # Process results for return
                processed_results = []
                for seq, prob, timestep_info in greedy_results:
                    # Convert indices to text
                    if len(seq) == 0:
                        text = ""
                        conf = prob
                    else:
                        char_list = [
                            self.character[idx]
                            for idx in seq
                            if idx < len(self.character)
                        ]
                        text = "".join(char_list)

                        if self.reverse:  # for arabic rec
                            text = self.pred_reverse(text)

                        conf = prob

                    result_item = {
                        "text": text,
                        "confidence": conf,
                        "info": timestep_info
                    }

                    processed_results.append(result_item)

                # Return with alignment info
                result = {"results": processed_results}

                if label is not None:
                    label_decoded = self.decode(label)
                    result["label"] = label_decoded

                return result
            else:
                # Original greedy decoding
                preds_idx = preds.argmax(axis=2)
                preds_prob = preds.max(axis=2)

        # Standard decoding path (original behavior)
        text = self.decode(
            preds_idx,
            preds_prob,
            is_remove_duplicate=True,
            return_word_box=return_word_box,
        )

        if return_word_box:
            for rec_idx, rec in enumerate(text):
                wh_ratio = kwargs.get("wh_ratio_list", [1.0] * len(text))[rec_idx]
                max_wh_ratio = kwargs.get("max_wh_ratio", 1.0)
                rec[2][0] = rec[2][0] * (wh_ratio / max_wh_ratio)

        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ["blank"] + dict_character
        return dict_character


class CTCLabelDecodeWithUnk(CTCLabelDecode):
    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super(CTCLabelDecodeWithUnk, self).__init__(
            character_dict_path, use_space_char, **kwargs
        )

    def add_special_char(self, dict_character):
        return super().add_special_char(dict_character) + ["unk"]


class DistillationCTCLabelDecode(CTCLabelDecode):
    """
    Convert
    Convert between text-label and text-index
    """

    def __init__(
        self,
        character_dict_path=None,
        use_space_char=False,
        model_name=["student"],
        key=None,
        multi_head=False,
        **kwargs,
    ):
        super(DistillationCTCLabelDecode, self).__init__(
            character_dict_path, use_space_char
        )
        if not isinstance(model_name, list):
            model_name = [model_name]
        self.model_name = model_name

        self.key = key
        self.multi_head = multi_head

    def __call__(self, preds, label=None, *args, **kwargs):
        output = dict()
        for name in self.model_name:
            pred = preds[name]
            if self.key is not None:
                pred = pred[self.key]
            if self.multi_head and isinstance(pred, dict):
                pred = pred["ctc"]
            output[name] = super().__call__(pred, label=label, *args, **kwargs)
        return output


class AttnLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super(AttnLabelDecode, self).__init__(character_dict_path, use_space_char)

    def add_special_char(self, dict_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = dict_character
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        [beg_idx, end_idx] = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if int(text_index[batch_idx][idx]) == int(end_idx):
                    break
                if is_remove_duplicate:
                    # only for predict
                    if (
                        idx > 0
                        and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]
                    ):
                        continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = "".join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def __call__(self, preds, label=None, *args, **kwargs):
        """
        text = self.decode(text)
        if label is None:
            return text
        else:
            label = self.decode(label, is_remove_duplicate=False)
            return text, label
        """
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()

        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        if label is None:
            return text
        label = self.decode(label, is_remove_duplicate=False)
        return text, label

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "unsupported type %s in get_beg_end_flag_idx" % beg_or_end
        return idx


class RFLLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super(RFLLabelDecode, self).__init__(character_dict_path, use_space_char)

    def add_special_char(self, dict_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = dict_character
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        [beg_idx, end_idx] = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if int(text_index[batch_idx][idx]) == int(end_idx):
                    break
                if is_remove_duplicate:
                    # only for predict
                    if (
                        idx > 0
                        and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]
                    ):
                        continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = "".join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def __call__(self, preds, label=None, *args, **kwargs):
        # if seq_outputs is not None:
        if isinstance(preds, tuple) or isinstance(preds, list):
            cnt_outputs, seq_outputs = preds
            if isinstance(seq_outputs, paddle.Tensor):
                seq_outputs = seq_outputs.numpy()
            preds_idx = seq_outputs.argmax(axis=2)
            preds_prob = seq_outputs.max(axis=2)
            text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)

            if label is None:
                return text
            label = self.decode(label, is_remove_duplicate=False)
            return text, label

        else:
            cnt_outputs = preds
            if isinstance(cnt_outputs, paddle.Tensor):
                cnt_outputs = cnt_outputs.numpy()
            cnt_length = []
            for lens in cnt_outputs:
                length = round(np.sum(lens))
                cnt_length.append(length)
            if label is None:
                return cnt_length
            label = self.decode(label, is_remove_duplicate=False)
            length = [len(res[0]) for res in label]
            return cnt_length, length

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "unsupported type %s in get_beg_end_flag_idx" % beg_or_end
        return idx


class SEEDLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super(SEEDLabelDecode, self).__init__(character_dict_path, use_space_char)

    def add_special_char(self, dict_character):
        self.padding_str = "padding"
        self.end_str = "eos"
        self.unknown = "unknown"
        dict_character = dict_character + [self.end_str, self.padding_str, self.unknown]
        return dict_character

    def get_ignored_tokens(self):
        end_idx = self.get_beg_end_flag_idx("eos")
        return [end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "sos":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "eos":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "unsupported type %s in get_beg_end_flag_idx" % beg_or_end
        return idx

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """convert text-index into text-label."""
        result_list = []
        [end_idx] = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if int(text_index[batch_idx][idx]) == int(end_idx):
                    break
                if is_remove_duplicate:
                    # only for predict
                    if (
                        idx > 0
                        and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]
                    ):
                        continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = "".join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def __call__(self, preds, label=None, *args, **kwargs):
        """
        text = self.decode(text)
        if label is None:
            return text
        else:
            label = self.decode(label, is_remove_duplicate=False)
            return text, label
        """
        preds_idx = preds["rec_pred"]
        if isinstance(preds_idx, paddle.Tensor):
            preds_idx = preds_idx.numpy()
        if "rec_pred_scores" in preds:
            preds_idx = preds["rec_pred"]
            preds_prob = preds["rec_pred_scores"]
        else:
            preds_idx = preds["rec_pred"].argmax(axis=2)
            preds_prob = preds["rec_pred"].max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        if label is None:
            return text
        label = self.decode(label, is_remove_duplicate=False)
        return text, label


class SRNLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super(SRNLabelDecode, self).__init__(character_dict_path, use_space_char)
        self.max_text_length = kwargs.get("max_text_length", 25)

    def __call__(self, preds, label=None, *args, **kwargs):
        pred = preds["predict"]
        char_num = len(self.character_str) + 2
        if isinstance(pred, paddle.Tensor):
            pred = pred.numpy()
        pred = np.reshape(pred, [-1, char_num])

        preds_idx = np.argmax(pred, axis=1)
        preds_prob = np.max(pred, axis=1)

        preds_idx = np.reshape(preds_idx, [-1, self.max_text_length])

        preds_prob = np.reshape(preds_prob, [-1, self.max_text_length])

        text = self.decode(preds_idx, preds_prob)

        if label is None:
            text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
            return text
        label = self.decode(label)
        return text, label

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)

        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if (
                        idx > 0
                        and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]
                    ):
                        continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)

            text = "".join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def add_special_char(self, dict_character):
        dict_character = dict_character + [self.beg_str, self.end_str]
        return dict_character

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "unsupported type %s in get_beg_end_flag_idx" % beg_or_end
        return idx


class ParseQLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index"""

    BOS = "[B]"
    EOS = "[E]"
    PAD = "[P]"

    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super(ParseQLabelDecode, self).__init__(character_dict_path, use_space_char)
        self.max_text_length = kwargs.get("max_text_length", 25)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, dict):
            pred = preds["predict"]
        else:
            pred = preds

        char_num = (
            len(self.character_str) + 1
        )  # We don't predict <bos> nor <pad>, with only addition <eos>
        if isinstance(pred, paddle.Tensor):
            pred = pred.numpy()
        B, L = pred.shape[:2]
        pred = np.reshape(pred, [-1, char_num])

        preds_idx = np.argmax(pred, axis=1)
        preds_prob = np.max(pred, axis=1)

        preds_idx = np.reshape(preds_idx, [B, L])
        preds_prob = np.reshape(preds_prob, [B, L])

        if label is None:
            text = self.decode(preds_idx, preds_prob, raw=False)
            return text

        text = self.decode(preds_idx, preds_prob, raw=False)
        label = self.decode(label, None, False)

        return text, label

    def decode(self, text_index, text_prob=None, raw=False):
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)

        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []

            index = text_index[batch_idx, :]
            prob = None
            if text_prob is not None:
                prob = text_prob[batch_idx, :]

            if not raw:
                index, prob = self._filter(index, prob)

            for idx in range(len(index)):
                if index[idx] in ignored_tokens:
                    continue
                char_list.append(self.character[int(index[idx])])
                if text_prob is not None:
                    conf_list.append(prob[idx])
                else:
                    conf_list.append(1)

            text = "".join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))

        return result_list

    def add_special_char(self, dict_character):
        dict_character = [self.EOS] + dict_character + [self.BOS, self.PAD]
        return dict_character

    def _filter(self, ids, probs=None):
        ids = ids.tolist()
        try:
            eos_idx = ids.index(self.dict[self.EOS])
        except ValueError:
            eos_idx = len(ids)  # Nothing to truncate.
        # Truncate after EOS
        ids = ids[:eos_idx]
        if probs is not None:
            probs = probs[: eos_idx + 1]  # but include prob. for EOS (if it exists)
        return ids, probs

    def get_ignored_tokens(self):
        return [self.dict[self.BOS], self.dict[self.EOS], self.dict[self.PAD]]


class SARLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super(SARLabelDecode, self).__init__(character_dict_path, use_space_char)

        self.rm_symbol = kwargs.get("rm_symbol", False)

    def add_special_char(self, dict_character):
        beg_end_str = "<BOS/EOS>"
        unknown_str = "<UKN>"
        padding_str = "<PAD>"
        dict_character = dict_character + [unknown_str]
        self.unknown_idx = len(dict_character) - 1
        dict_character = dict_character + [beg_end_str]
        self.start_idx = len(dict_character) - 1
        self.end_idx = len(dict_character) - 1
        dict_character = dict_character + [padding_str]
        self.padding_idx = len(dict_character) - 1
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()

        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if int(text_index[batch_idx][idx]) == int(self.end_idx):
                    if text_prob is None and idx == 0:
                        continue
                    else:
                        break
                if is_remove_duplicate:
                    # only for predict
                    if (
                        idx > 0
                        and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]
                    ):
                        continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = "".join(char_list)
            if self.rm_symbol:
                comp = re.compile("[^A-Z^a-z^0-9^\u4e00-\u9fa5]")
                text = text.lower()
                text = comp.sub("", text)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)

        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)

        if label is None:
            return text
        label = self.decode(label, is_remove_duplicate=False)
        return text, label

    def get_ignored_tokens(self):
        return [self.padding_idx]


class SATRNLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super(SATRNLabelDecode, self).__init__(character_dict_path, use_space_char)

        self.rm_symbol = kwargs.get("rm_symbol", False)

    def add_special_char(self, dict_character):
        beg_end_str = "<BOS/EOS>"
        unknown_str = "<UKN>"
        padding_str = "<PAD>"
        dict_character = dict_character + [unknown_str]
        self.unknown_idx = len(dict_character) - 1
        dict_character = dict_character + [beg_end_str]
        self.start_idx = len(dict_character) - 1
        self.end_idx = len(dict_character) - 1
        dict_character = dict_character + [padding_str]
        self.padding_idx = len(dict_character) - 1
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()

        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if int(text_index[batch_idx][idx]) == int(self.end_idx):
                    if text_prob is None and idx == 0:
                        continue
                    else:
                        break
                if is_remove_duplicate:
                    # only for predict
                    if (
                        idx > 0
                        and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]
                    ):
                        continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = "".join(char_list)
            if self.rm_symbol:
                comp = re.compile("[^A-Z^a-z^0-9^\u4e00-\u9fa5]")
                text = text.lower()
                text = comp.sub("", text)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)

        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)

        if label is None:
            return text
        label = self.decode(label, is_remove_duplicate=False)
        return text, label

    def get_ignored_tokens(self):
        return [self.padding_idx]


class DistillationSARLabelDecode(SARLabelDecode):
    """
    Convert
    Convert between text-label and text-index
    """

    def __init__(
        self,
        character_dict_path=None,
        use_space_char=False,
        model_name=["student"],
        key=None,
        multi_head=False,
        **kwargs,
    ):
        super(DistillationSARLabelDecode, self).__init__(
            character_dict_path, use_space_char
        )
        if not isinstance(model_name, list):
            model_name = [model_name]
        self.model_name = model_name

        self.key = key
        self.multi_head = multi_head

    def __call__(self, preds, label=None, *args, **kwargs):
        output = dict()
        for name in self.model_name:
            pred = preds[name]
            if self.key is not None:
                pred = pred[self.key]
            if self.multi_head and isinstance(pred, dict):
                pred = pred["sar"]
            output[name] = super().__call__(pred, label=label, *args, **kwargs)
        return output


class PRENLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super(PRENLabelDecode, self).__init__(character_dict_path, use_space_char)

    def add_special_char(self, dict_character):
        padding_str = "<PAD>"  # 0
        end_str = "<EOS>"  # 1
        unknown_str = "<UNK>"  # 2

        dict_character = [padding_str, end_str, unknown_str] + dict_character
        self.padding_idx = 0
        self.end_idx = 1
        self.unknown_idx = 2

        return dict_character

    def decode(self, text_index, text_prob=None):
        """convert text-index into text-label."""
        result_list = []
        batch_size = len(text_index)

        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] == self.end_idx:
                    break
                if text_index[batch_idx][idx] in [self.padding_idx, self.unknown_idx]:
                    continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)

            text = "".join(char_list)
            if len(text) > 0:
                result_list.append((text, np.mean(conf_list).tolist()))
            else:
                # here confidence of empty recog result is 1
                result_list.append(("", 1))
        return result_list

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob)
        if label is None:
            return text
        label = self.decode(label)
        return text, label


class NRTRLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, use_space_char=True, **kwargs):
        super(NRTRLabelDecode, self).__init__(character_dict_path, use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if len(preds) == 2:
            preds_id = preds[0]
            preds_prob = preds[1]
            if isinstance(preds_id, paddle.Tensor):
                preds_id = preds_id.numpy()
            if isinstance(preds_prob, paddle.Tensor):
                preds_prob = preds_prob.numpy()
            if preds_id[0][0] == 2:
                preds_idx = preds_id[:, 1:]
                preds_prob = preds_prob[:, 1:]
            else:
                preds_idx = preds_id
            text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
            if label is None:
                return text
            label = self.decode(label[:, 1:])
        elif len(preds) == 4:
            preds_id = preds[0]
            preds_prob = preds[1]
            candidates_indices = preds[2]
            candidates_probs = preds[3]
            if isinstance(preds_id, paddle.Tensor):
                preds_id = preds_id.numpy()
            if isinstance(preds_prob, paddle.Tensor):
                preds_prob = preds_prob.numpy()
            if isinstance(candidates_indices, paddle.Tensor):
                candidates_indices = candidates_indices.numpy()
            if isinstance(candidates_probs, paddle.Tensor):
                candidates_probs = candidates_probs.numpy()
            if preds_id[0][0] == 2:
                preds_idx = preds_id[:, 1:]
                preds_prob = preds_prob[:, 1:]
                candidates_indices = candidates_indices[:, :, :]
                candidates_probs = candidates_probs[:, :, :]
            else:
                preds_idx = preds_id
            text = self.decode(preds_idx, preds_prob, candidates_indices, candidates_probs, is_remove_duplicate=False)
            if label is None:
                return text
            label = self.decode(label[:, 1:])
            return text, label
        else:
            if isinstance(preds, paddle.Tensor):
                preds = preds.numpy()
            preds_idx = preds.argmax(axis=2)
            preds_prob = preds.max(axis=2)
            text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
            if label is None:
                return text
            label = self.decode(label[:, 1:])
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ["blank", "<unk>", "<s>", "</s>"] + dict_character
        return dict_character

    def decode(self, text_index, text_prob=None, candidates_indices=None, candidates_probs=None, is_remove_duplicate=False):
        """Convert text-index into text-label."""
        result_list = []
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            candidates_list = []
            for idx in range(len(text_index[batch_idx])):
                try:
                    char_idx = self.character[int(text_index[batch_idx][idx])]
                except:
                    continue
                if char_idx == "</s>":  # end
                    break
                char_list.append(char_idx)
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
                if candidates_indices is not None and candidates_probs is not None:
                    topk_chars = [self.character[int(cand)] for cand in candidates_indices[batch_idx, idx]]
                    topk_probs = candidates_probs[batch_idx, idx]
                    candidates_list.append(list(zip(topk_chars, topk_probs)))
            text = "".join(char_list)
            mean_conf = np.mean(conf_list).tolist() if conf_list else 1.0
            if candidates_list:
                result_list.append((text, mean_conf, candidates_list))
            else:
                result_list.append((text, mean_conf))
        return result_list


class MultiHeadLabelDecode(object):
    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        self.decode_list = kwargs.pop("decoder_list")
        self.gtc_decoder = "sar"
        for idx, decode_name in enumerate(self.decode_list):
            name = list(decode_name)[0]
            if name == "SARLabelDecode":
                # sar head
                sar_args = self.decode_list[idx][name]
                if sar_args is not None:
                    kwargs.update(sar_args)
                self.sar_decoder = SARLabelDecode(character_dict_path, use_space_char, **kwargs)
            elif name == "NRTRLabelDecode":
                gtc_args = self.decode_list[idx][name]
                if gtc_args is not None:
                    kwargs.update(gtc_args)
                self.gtc_decoder = NRTRLabelDecode(character_dict_path, use_space_char, **kwargs)
            elif name == "CTCLabelDecode":
                ctc_args = self.decode_list[idx][name]
                if ctc_args is not None:
                    kwargs.update(ctc_args)
                self.ctc_decoder = CTCLabelDecode(character_dict_path, use_space_char, **kwargs)
            elif name == "BeamCTCLabelDecode":
                ctc_args = self.decode_list[idx][name]
                if ctc_args is not None:
                    kwargs.update(ctc_args)
                self.ctc_decoder = BeamCTCLabelDecode(character_dict_path, use_space_char, **kwargs)
            else:
                raise ValueError(f"{name} is not supported in MultiHeadLabelDecode")
        if hasattr(self, "ctc_decoder"):
            self.character = getattr(self.ctc_decoder, "character")

    def __call__(self, head_out, **kwargs):
        # Currently, MultiHeadLabelDecode only supports test-time decoding
        ctc_logits = head_out["ctc"]
        ctc_args, gtc_args, sar_agrs = kwargs.get("ctc", {}), kwargs.get("gtc", {}), kwargs.get("sar", {})
        results = dict()
        results["ctc"] = self.ctc_decoder(ctc_logits, **ctc_args)
        if self.gtc_decoder == "sar":
            results["sar"] = self.sar_decoder(head_out["sar"], **sar_agrs)
        else:
            results["gtc"] = self.gtc_decoder(head_out["gtc"], **gtc_args)
        return results


class ViTSTRLabelDecode(NRTRLabelDecode):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super(ViTSTRLabelDecode, self).__init__(character_dict_path, use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, paddle.Tensor):
            preds = preds[:, 1:].numpy()
        else:
            preds = preds[:, 1:]
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        if label is None:
            return text
        label = self.decode(label[:, 1:])
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ["<s>", "</s>"] + dict_character
        return dict_character


class ABINetLabelDecode(NRTRLabelDecode):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super(ABINetLabelDecode, self).__init__(character_dict_path, use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, dict):
            preds = preds["align"][-1].numpy()
        elif isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        else:
            preds = preds

        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ["</s>"] + dict_character
        return dict_character


class SPINLabelDecode(AttnLabelDecode):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super(SPINLabelDecode, self).__init__(character_dict_path, use_space_char)

    def add_special_char(self, dict_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = dict_character
        dict_character = [self.beg_str] + [self.end_str] + dict_character
        return dict_character


class VLLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super(VLLabelDecode, self).__init__(character_dict_path, use_space_char)
        self.max_text_length = kwargs.get("max_text_length", 25)
        self.nclass = len(self.character) + 1

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id - 1]
                for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = "".join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def __call__(self, preds, label=None, length=None, *args, **kwargs):
        if len(preds) == 2:  # eval mode
            text_pre, x = preds
            b = text_pre.shape[1]
            lenText = self.max_text_length
            nsteps = self.max_text_length

            if not isinstance(text_pre, paddle.Tensor):
                text_pre = paddle.to_tensor(text_pre, dtype="float32")

            out_res = paddle.zeros(shape=[lenText, b, self.nclass], dtype=x.dtype)
            out_length = paddle.zeros(shape=[b], dtype=x.dtype)
            now_step = 0
            for _ in range(nsteps):
                if 0 in out_length and now_step < nsteps:
                    tmp_result = text_pre[now_step, :, :]
                    out_res[now_step] = tmp_result
                    tmp_result = tmp_result.topk(1)[1].squeeze(axis=1)
                    for j in range(b):
                        if out_length[j] == 0 and tmp_result[j] == 0:
                            out_length[j] = now_step + 1
                    now_step += 1
            for j in range(0, b):
                if int(out_length[j]) == 0:
                    out_length[j] = nsteps
            start = 0
            output = paddle.zeros(
                shape=[int(out_length.sum()), self.nclass], dtype=x.dtype
            )
            for i in range(0, b):
                cur_length = int(out_length[i])
                output[start : start + cur_length] = out_res[0:cur_length, i, :]
                start += cur_length
            net_out = output
            length = out_length

        else:  # train mode
            net_out = preds[0]
            length = length
            net_out = paddle.concat([t[:l] for t, l in zip(net_out, length)])
        text = []
        if not isinstance(net_out, paddle.Tensor):
            net_out = paddle.to_tensor(net_out, dtype="float32")
        net_out = F.softmax(net_out, axis=1)
        for i in range(0, length.shape[0]):
            if i == 0:
                start_idx = 0
                end_idx = int(length[i])
            else:
                start_idx = int(length[:i].sum())
                end_idx = int(length[:i].sum() + length[i])
            preds_idx = net_out[start_idx:end_idx].topk(1)[1][:, 0].tolist()
            preds_text = "".join(
                [
                    (
                        self.character[idx - 1]
                        if idx > 0 and idx <= len(self.character)
                        else ""
                    )
                    for idx in preds_idx
                ]
            )
            preds_prob = net_out[start_idx:end_idx].topk(1)[0][:, 0]
            preds_prob = paddle.exp(
                paddle.log(preds_prob).sum() / (preds_prob.shape[0] + 1e-6)
            )
            text.append((preds_text, float(preds_prob)))
        if label is None:
            return text
        label = self.decode(label)
        return text, label


class CANLabelDecode(BaseRecLabelDecode):
    """Convert between latex-symbol and symbol-index"""

    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super(CANLabelDecode, self).__init__(character_dict_path, use_space_char)

    def decode(self, text_index, preds_prob=None):
        result_list = []
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            seq_end = text_index[batch_idx].argmin(0)
            idx_list = text_index[batch_idx][:seq_end].tolist()
            symbol_list = [self.character[idx] for idx in idx_list]
            probs = []
            if preds_prob is not None:
                probs = preds_prob[batch_idx][: len(symbol_list)].tolist()

            result_list.append([" ".join(symbol_list), probs])
        return result_list

    def __call__(self, preds, label=None, *args, **kwargs):
        pred_prob, _, _, _ = preds
        preds_idx = pred_prob.argmax(axis=2)

        text = self.decode(preds_idx)
        if label is None:
            return text
        label = self.decode(label)
        return text, label


class CPPDLabelDecode(NRTRLabelDecode):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super(CPPDLabelDecode, self).__init__(character_dict_path, use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, tuple):
            if isinstance(preds[-1], dict):
                preds = preds[-1]["align"][-1].numpy()
            else:
                preds = preds[-1].numpy()
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        else:
            preds = preds
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ["</s>"] + dict_character
        return dict_character


class LaTeXOCRDecode(object):
    """Convert between latex-symbol and symbol-index"""

    def __init__(self, rec_char_dict_path, **kwargs):
        from tokenizers import Tokenizer as TokenizerFast

        super(LaTeXOCRDecode, self).__init__()
        self.tokenizer = TokenizerFast.from_file(rec_char_dict_path)

    def post_process(self, s):
        text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
        letter = "[a-zA-Z]"
        noletter = "[\W_^\d]"
        names = [x[0].replace(" ", "") for x in re.findall(text_reg, s)]
        s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
        news = s
        while True:
            s = news
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", s)
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
            news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
            if news == s:
                break
        return s

    def decode(self, tokens):
        if len(tokens.shape) == 1:
            tokens = tokens[None, :]
        dec = [self.tokenizer.decode(tok) for tok in tokens]
        dec_str_list = [
            "".join(detok.split(" "))
            .replace("Ġ", " ")
            .replace("[EOS]", "")
            .replace("[BOS]", "")
            .replace("[PAD]", "")
            .strip()
            for detok in dec
        ]
        return [self.post_process(dec_str) for dec_str in dec_str_list]

    def __call__(self, preds, label=None, mode="eval", *args, **kwargs):
        if mode == "train":
            preds_idx = np.array(preds.argmax(axis=2))
            text = self.decode(preds_idx)
        else:
            text = self.decode(np.array(preds))
        if label is None:
            return text
        label = self.decode(np.array(label))
        return text, label


class UniMERNetDecode(object):
    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    ]

    def __init__(
        self,
        rec_char_dict_path,
        is_infer=False,
        **kwargs,
    ):
        from tokenizers import Tokenizer as TokenizerFast
        from tokenizers import AddedToken

        self.is_infer = is_infer
        self._unk_token = "<unk>"
        self._bos_token = "<s>"
        self._eos_token = "</s>"
        self._pad_token = "<pad>"
        self._sep_token = None
        self._cls_token = None
        self._mask_token = None
        self._additional_special_tokens = []
        self.model_input_names = ["input_ids", "token_type_ids", "attention_mask"]
        self.max_seq_len = 2048
        self.pad_token_id = 1
        self.bos_token_id = 0
        self.eos_token_id = 2
        self.padding_side = "right"
        self.pad_token_id = 1
        self.pad_token = "<pad>"
        self.pad_token_type_id = 0
        self.pad_to_multiple_of = None
        fast_tokenizer_file = os.path.join(rec_char_dict_path, "tokenizer.json")
        tokenizer_config_file = os.path.join(
            rec_char_dict_path, "tokenizer_config.json"
        )
        self.tokenizer = TokenizerFast.from_file(fast_tokenizer_file)
        added_tokens_decoder = {}
        added_tokens_map = {}
        if tokenizer_config_file is not None:
            with open(
                tokenizer_config_file, encoding="utf-8"
            ) as tokenizer_config_handle:
                init_kwargs = json.load(tokenizer_config_handle)
                if "added_tokens_decoder" in init_kwargs:
                    for idx, token in init_kwargs["added_tokens_decoder"].items():
                        if isinstance(token, dict):
                            token = AddedToken(**token)
                        if isinstance(token, AddedToken):
                            added_tokens_decoder[int(idx)] = token
                            added_tokens_map[str(token)] = token
                        else:
                            raise ValueError(
                                f"Found a {token.__class__} in the saved `added_tokens_decoder`, should be a dictionary or an AddedToken instance"
                            )
                init_kwargs["added_tokens_decoder"] = added_tokens_decoder
                added_tokens_decoder = init_kwargs.pop("added_tokens_decoder", {})
                tokens_to_add = [
                    token
                    for index, token in sorted(
                        added_tokens_decoder.items(), key=lambda x: x[0]
                    )
                    if token not in added_tokens_decoder
                ]
                added_tokens_encoder = self.added_tokens_encoder(added_tokens_decoder)
                encoder = list(added_tokens_encoder.keys()) + [
                    str(token) for token in tokens_to_add
                ]
                tokens_to_add += [
                    token
                    for token in self.all_special_tokens_extended
                    if token not in encoder and token not in tokens_to_add
                ]
                if len(tokens_to_add) > 0:
                    is_last_special = None
                    tokens = []
                    special_tokens = self.all_special_tokens
                    for token in tokens_to_add:
                        is_special = (
                            (token.special or str(token) in special_tokens)
                            if isinstance(token, AddedToken)
                            else str(token) in special_tokens
                        )
                        if is_last_special is None or is_last_special == is_special:
                            tokens.append(token)
                        else:
                            self._add_tokens(tokens, special_tokens=is_last_special)
                            tokens = [token]
                        is_last_special = is_special
                    if tokens:
                        self._add_tokens(tokens, special_tokens=is_last_special)

    def _add_tokens(self, new_tokens, special_tokens=False) -> int:
        if special_tokens:
            return self.tokenizer.add_special_tokens(new_tokens)

        return self.tokenizer.add_tokens(new_tokens)

    def added_tokens_encoder(self, added_tokens_decoder):
        return {
            k.content: v
            for v, k in sorted(added_tokens_decoder.items(), key=lambda item: item[0])
        }

    @property
    def all_special_tokens(self):
        all_toks = [str(s) for s in self.all_special_tokens_extended]
        return all_toks

    @property
    def all_special_tokens_extended(self):
        all_tokens = []
        seen = set()
        for value in self.special_tokens_map_extended.values():
            if isinstance(value, (list, tuple)):
                tokens_to_add = [token for token in value if str(token) not in seen]
            else:
                tokens_to_add = [value] if str(value) not in seen else []
            seen.update(map(str, tokens_to_add))
            all_tokens.extend(tokens_to_add)
        return all_tokens

    @property
    def special_tokens_map_extended(self):
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    def convert_ids_to_tokens(self, ids, skip_special_tokens: bool = False):
        if isinstance(ids, int):
            return self.tokenizer.id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            tokens.append(self.tokenizer.id_to_token(index))
        return tokens

    def detokenize(self, tokens):
        self.tokenizer.bos_token = "<s>"
        self.tokenizer.eos_token = "</s>"
        self.tokenizer.pad_token = "<pad>"
        toks = [self.convert_ids_to_tokens(tok) for tok in tokens]
        for b in range(len(toks)):
            for i in reversed(range(len(toks[b]))):
                if toks[b][i] is None:
                    toks[b][i] = ""
                toks[b][i] = toks[b][i].replace("Ġ", " ").strip()
                if toks[b][i] in (
                    [
                        self.tokenizer.bos_token,
                        self.tokenizer.eos_token,
                        self.tokenizer.pad_token,
                    ]
                ):
                    del toks[b][i]
        return toks

    def token2str(self, token_ids) -> list:
        generated_text = []
        for tok_id in token_ids:
            end_idx = np.argwhere(tok_id == 2)
            if len(end_idx) > 0:
                end_idx = int(end_idx[0][0])
                tok_id = tok_id[: end_idx + 1]
            generated_text.append(
                self.tokenizer.decode(tok_id, skip_special_tokens=True)
            )
        generated_text = [self.post_process(text) for text in generated_text]
        return generated_text

    def normalize_infer(self, s: str) -> str:
        """Normalizes a string by removing unnecessary spaces.

        Args:
            s (str): String to normalize.

        Returns:
            str: Normalized string.
        """
        text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
        letter = "[a-zA-Z]"
        noletter = "[\W_^\d]"
        names = []
        for x in re.findall(text_reg, s):
            pattern = r"\\[a-zA-Z]+"
            pattern = r"(\\[a-zA-Z]+)\s(?=\w)|\\[a-zA-Z]+\s(?=})"
            matches = re.findall(pattern, x[0])
            for m in matches:
                if (
                    m
                    not in [
                        "\\operatorname",
                        "\\mathrm",
                        "\\text",
                        "\\mathbf",
                    ]
                    and m.strip() != ""
                ):
                    s = s.replace(m, m + "XXXXXXX")
                    s = s.replace(" ", "")
                    names.append(s)
        if len(names) > 0:
            s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
        news = s
        while True:
            s = news
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", s)
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
            news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
            if news == s:
                break
        return s.replace("XXXXXXX", " ")

    def remove_chinese_text_wrapping(self, formula):
        pattern = re.compile(r"\\text\s*{\s*([^}]*?[\u4e00-\u9fff]+[^}]*?)\s*}")

        def replacer(match):
            return match.group(1)

        replaced_formula = pattern.sub(replacer, formula)
        return replaced_formula.replace('"', "")

    def normalize(self, s):
        text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
        letter = "[a-zA-Z]"
        noletter = "[\W_^\d]"
        names = [x[0].replace(" ", "") for x in re.findall(text_reg, s)]
        s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
        news = s
        while True:
            s = news
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", s)
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
            news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
            if news == s:
                break
        return s

    def post_process(self, text: str) -> str:
        """Post-processes a string by fixing text and normalizing it.

        Args:
            text (str): String to post-process.

        Returns:
            str: Post-processed string.
        """
        from ftfy import fix_text

        if self.is_infer:
            text = self.remove_chinese_text_wrapping(text)
            text = fix_text(text)
            text = self.normalize_infer(text)
        else:
            text = fix_text(text)
            text = self.normalize(text)
        return text

    def __call__(self, preds, label=None, mode="eval", *args, **kwargs):
        if mode == "train":
            preds_idx = np.array(preds.argmax(axis=2))
            text = self.token2str(preds_idx)
        else:
            text = self.token2str(np.array(preds))
        if label is None:
            return text
        label = self.token2str(np.array(label))
        return text, label
