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
import gc
import numpy as np
import io
import math
import traceback
import os
from paddle.io import Dataset
import lmdb
import cv2
import string
import pickle
from PIL import Image
from collections import defaultdict

from .imaug import transform, create_operators


class LMDBDataSet(Dataset):
    def __init__(self, config, mode, logger, seed=None, read_masks=False):
        super(LMDBDataSet, self).__init__()

        global_config = config["Global"]
        dataset_config = config[mode]["dataset"]
        loader_config = config[mode]["loader"]
        loader_config["batch_size_per_card"]
        data_dir = dataset_config["data_dir"]
        self.do_shuffle = loader_config["shuffle"]
        self.seed = seed
        self.read_masks = read_masks
        self.lmdb_sets = self.load_hierarchical_lmdb_dataset(data_dir)
        logger.info("Initialize indexes of datasets:%s" % data_dir)
        self.data_idx_order_list = self.dataset_traversal()
        if self.do_shuffle:
            np.random.shuffle(self.data_idx_order_list)
        self.ops = create_operators(dataset_config["transforms"], global_config)
        self.ext_op_transform_idx = dataset_config.get("ext_op_transform_idx", 1)

        ratio_list = dataset_config.get("ratio_list", [1.0])
        self.need_reset = True in [x < 1 for x in ratio_list]

    def load_hierarchical_lmdb_dataset(self, data_dir):
        lmdb_sets = {}
        dataset_idx = 0
        for dirpath, dirnames, filenames in os.walk(data_dir + "/"):
            if not dirnames:
                env = lmdb.open(
                    dirpath,
                    max_readers=32,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )
                txn = env.begin(write=False)
                num_samples = int(txn.get("num-samples".encode()))
                lmdb_sets[dataset_idx] = {
                    "dirpath": dirpath,
                    "env": env,
                    "txn": txn,
                    "num_samples": num_samples,
                }
                dataset_idx += 1
        return lmdb_sets

    def dataset_traversal(self):
        lmdb_num = len(self.lmdb_sets)
        total_sample_num = 0
        for lno in range(lmdb_num):
            total_sample_num += self.lmdb_sets[lno]["num_samples"]
        data_idx_order_list = np.zeros((total_sample_num, 2))
        beg_idx = 0
        for lno in range(lmdb_num):
            tmp_sample_num = self.lmdb_sets[lno]["num_samples"]
            end_idx = beg_idx + tmp_sample_num
            data_idx_order_list[beg_idx:end_idx, 0] = lno
            data_idx_order_list[beg_idx:end_idx, 1] = list(range(tmp_sample_num))
            data_idx_order_list[beg_idx:end_idx, 1] += 1
            beg_idx = beg_idx + tmp_sample_num
        return data_idx_order_list

    def get_img_data(self, value):
        """get_img_data"""
        if not value:
            return None
        imgdata = np.frombuffer(value, dtype="uint8")
        if imgdata is None:
            return None
        imgori = cv2.imdecode(imgdata, 1)
        if imgori is None:
            return None
        return imgori

    def get_masks_data(self, txn, index):
        meta_key = "meta-%09d".encode() % index
        meta = txn.get(meta_key)
        if not meta:
            return None
        masks = meta[4]
        return masks

    def get_ext_data(self):
        ext_data_num = 0
        for op in self.ops:
            if hasattr(op, "ext_data_num"):
                ext_data_num = getattr(op, "ext_data_num")
                break
        load_data_ops = self.ops[: self.ext_op_transform_idx]
        ext_data = []

        while len(ext_data) < ext_data_num:
            lmdb_idx, file_idx = self.data_idx_order_list[np.random.randint(len(self))]
            lmdb_idx = int(lmdb_idx)
            file_idx = int(file_idx)
            sample_info = self.get_lmdb_sample_info(
                self.lmdb_sets[lmdb_idx]["txn"], file_idx
            )
            if sample_info is None:
                continue
            img, label = sample_info
            data = {"image": img, "label": label}
            if self.read_masks:
                data["masks"] = self.get_masks_data(self.lmdb_sets[lmdb_idx]["txn"], file_idx)
            data = transform(data, load_data_ops)
            if data is None:
                continue
            ext_data.append(data)
        return ext_data

    def get_lmdb_sample_info(self, txn, index):
        label_key = "label-%09d".encode() % index
        label = txn.get(label_key)
        if label is None:
            return None
        label = label.decode("utf-8")
        img_key = "image-%09d".encode() % index
        imgbuf = txn.get(img_key)
        return imgbuf, label

    def __getitem__(self, idx):
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        sample_info = self.get_lmdb_sample_info(
            self.lmdb_sets[lmdb_idx]["txn"], file_idx
        )
        if sample_info is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        img, label = sample_info
        data = {"image": img, "label": label}
        data["ext_data"] = self.get_ext_data()
        if self.read_masks:
            data["masks"] = self.get_masks_data(
                self.lmdb_sets[lmdb_idx]["txn"], file_idx
            )
        outs = transform(data, self.ops)
        if outs is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        return outs

    def __len__(self):
        return self.data_idx_order_list.shape[0]


class CurriculumLMDBDataSet(Dataset):
    def __init__(self, config, mode, logger, seed=None, read_masks=False):
        super(CurriculumLMDBDataSet, self).__init__()
        self.current_epoch = 0
        global_config = config["Global"]
        dataset_config = config[mode]["dataset"]
        loader_config = config[mode]["loader"]
        data_dir = dataset_config["data_dir"]
        self.logger = logger
        self.length_steps = dataset_config.get("length_steps", [25, 30, 40])
        self.stage_epochs = dataset_config.get("stage_epochs", [5, 5])
        assert len(self.stage_epochs) - 1 == len(self.length_steps), (
            "stage_epochs must less than length_steps by 1 unit"
        )
        self.do_shuffle = loader_config["shuffle"]
        self.seed = seed
        self.read_masks = read_masks
        self.lmdb_sets = self.load_hierarchical_lmdb_dataset(data_dir)
        self.logger.info("Initialize indexs of datasets:%s" % data_dir)
        self._prepare_stages()
        self.data_idx_order_list = self.dataset_traversal()
        if self.do_shuffle:
            np.random.shuffle(self.data_idx_order_list)
        self.ops = create_operators(dataset_config["transforms"], global_config)
        self.ext_op_transform_idx = dataset_config.get("ext_op_transform_idx", 1)
        ratio_list = dataset_config.get("ratio_list", [1.0])
        self.need_reset = True in [x < 1 for x in ratio_list]

    def load_hierarchical_lmdb_dataset(self, data_dir):
        lmdb_sets = {}
        dataset_idx = 0
        for dirpath, dirnames, filenames in os.walk(data_dir + "/"):
            if not dirnames:
                env = lmdb.open(
                    dirpath,
                    max_readers=32,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )
                txn = env.begin(write=False)
                num_samples = int(txn.get("num-samples".encode()))
                lmdb_sets[dataset_idx] = {
                    "dirpath": dirpath,
                    "env": env,
                    "txn": txn,
                    "num_samples": num_samples,
                }
                dataset_idx += 1
        return lmdb_sets

    def _identify_stage(self):
        return np.searchsorted(self.cum_epochs, self.current_epoch, side="right")

    def _prepare_stages(self):
        self.cum_epochs = np.cumsum(self.stage_epochs)
        self.stage = -1
        lmdb_num = len(self.lmdb_sets)
        for lno in range(lmdb_num):
            tmp_sample_num = self.lmdb_sets[lno]["num_samples"]
            indices_per_length = defaultdict(list)
            for idx in range(1, tmp_sample_num + 1):
                length = self.get_lmdb_sample_length(self.lmdb_sets[lno]["txn"], idx)
                if not length:
                    continue
                for length_step in self.length_steps:
                    if length > length_step:
                        continue
                    indices_per_length[length_step].append(idx)
                    break
            self.lmdb_sets[lno]["indices_per_length"] = indices_per_length
        total_indices_per_length = defaultdict(int)
        for dataset_idx, dataset_info in self.lmdb_sets.items():
            for length_step in self.length_steps:
                total_indices_per_length[length_step] += len(
                    dataset_info["indices_per_length"][length_step]
                )

        self.logger.info(
            f"Total sum of indices_per_length across all datasets: "
            f"{', '.join([f'{length_step}: {total_indices_per_length[length_step]}' for length_step in self.length_steps])}"
        )

    def get_lmdb_sample_length(self, txn, index):
        label_key = "label-%09d".encode() % index
        label = txn.get(label_key)
        if label is None:
            return None
        return len(label.decode("utf-8"))

    def dataset_traversal(self):
        lmdb_num = len(self.lmdb_sets)
        stage = self._identify_stage()
        if stage == self.stage:
            return self.data_idx_order_list
        self.logger.info(
            f"Transitioning to stage {stage} at epoch {self.current_epoch}"
        )
        self.stage = stage
        max_len_idx = min(self.stage, len(self.length_steps) - 1)
        total_sample_num = 0
        for lno in range(lmdb_num):
            for idx in range(0, max_len_idx + 1):
                total_sample_num += len(
                    self.lmdb_sets[lno]["indices_per_length"][self.length_steps[idx]]
                )
        self.logger.info(
            f"Total samples at epoch {self.current_epoch}: {total_sample_num}"
        )
        data_idx_order_list = np.zeros((total_sample_num, 2))
        beg_idx = 0
        for lno in range(lmdb_num):
            tmp_sample_num = 0
            tmp_indices = []
            for idx in range(0, max_len_idx + 1):
                tmp_sample_num += len(
                    self.lmdb_sets[lno]["indices_per_length"][self.length_steps[idx]]
                )
                tmp_indices.extend(
                    self.lmdb_sets[lno]["indices_per_length"][self.length_steps[idx]]
                )
            end_idx = beg_idx + tmp_sample_num
            data_idx_order_list[beg_idx:end_idx, 0] = lno
            data_idx_order_list[beg_idx:end_idx, 1] = tmp_indices
            beg_idx = beg_idx + tmp_sample_num
        return data_idx_order_list

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        self.data_idx_order_list = self.dataset_traversal()

    def get_img_data(self, value):
        """get_img_data"""
        if not value:
            return None
        imgdata = np.frombuffer(value, dtype="uint8")
        if imgdata is None:
            return None
        imgori = cv2.imdecode(imgdata, 1)
        if imgori is None:
            return None
        return imgori

    def get_masks_data(self, txn, index):
        meta_key = "meta-%09d".encode() % index
        meta = txn.get(meta_key)
        if not meta:
            return None
        masks = meta[4]
        return masks

    def get_ext_data(self):
        ext_data_num = 0
        for op in self.ops:
            if hasattr(op, "ext_data_num"):
                ext_data_num = getattr(op, "ext_data_num")
                break
        load_data_ops = self.ops[: self.ext_op_transform_idx]
        ext_data = []

        while len(ext_data) < ext_data_num:
            lmdb_idx, file_idx = self.data_idx_order_list[np.random.randint(len(self))]
            lmdb_idx = int(lmdb_idx)
            file_idx = int(file_idx)
            sample_info = self.get_lmdb_sample_info(
                self.lmdb_sets[lmdb_idx]["txn"], file_idx
            )
            if sample_info is None:
                continue
            img, label = sample_info
            data = {"image": img, "label": label}
            if self.read_masks:
                data["masks"] = self.get_masks_data(self.lmdb_sets[lmdb_idx]["txn"], file_idx)
            data = transform(data, load_data_ops)
            if data is None:
                continue
            ext_data.append(data)
        return ext_data

    def get_lmdb_sample_info(self, txn, index):
        label_key = "label-%09d".encode() % index
        label = txn.get(label_key)
        if label is None:
            return None
        label = label.decode("utf-8")
        img_key = "image-%09d".encode() % index
        imgbuf = txn.get(img_key)
        return imgbuf, label

    def __getitem__(self, idx):
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        sample_info = self.get_lmdb_sample_info(
            self.lmdb_sets[lmdb_idx]["txn"], file_idx
        )
        if sample_info is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        img, label = sample_info
        data = {"image": img, "label": label}
        data["ext_data"] = self.get_ext_data()
        if self.read_masks:
            data["masks"] = self.get_masks_data(self.lmdb_sets[lmdb_idx]["txn"], file_idx)
        outs = transform(data, self.ops)
        if outs is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        return outs

    def __len__(self):
        return self.data_idx_order_list.shape[0]


class MultiScaleLMDBDataSet(LMDBDataSet):
    def __init__(self, config, mode, logger, seed=None, read_masks=False):
        super(MultiScaleLMDBDataSet, self).__init__(config, mode, logger, seed, read_masks)
        self.logger = logger

        # Get dataset config
        dataset_config = config[mode]["dataset"]

        # Initialize variables for width-height ratio awareness
        self.ds_width = dataset_config.get("ds_width", False)

        # If using dynamic width, prepare ratios
        if self.ds_width:
            self.prepare_wh_ratio_data()

    def prepare_wh_ratio_data(self):
        """
        Prepare width-height ratio data for dynamic scaling
        This needs to collect ratio information from all LMDB entries
        """
        self.wh_ratio = []
        self.wh_ratio_idx_map = []

        # Iterate through all LMDB sets
        for lmdb_idx in range(len(self.lmdb_sets)):
            txn = self.lmdb_sets[lmdb_idx]["txn"]
            num_samples = self.lmdb_sets[lmdb_idx]["num_samples"]

            # Collect width-height ratios from all samples
            for idx in range(1, num_samples + 1):
                sample_info = self.get_lmdb_sample_info(txn, idx)
                if sample_info is None:
                    continue

                img_buf, label = sample_info

                # Get image dimensions
                try:
                    imgdata = np.frombuffer(img_buf, dtype="uint8")
                    img = cv2.imdecode(imgdata, 1)
                    h, w = img.shape[:2]
                    ratio = float(w) / float(h)

                    del img_buf, img
                    gc.collect()

                    # Store the ratio and corresponding index
                    self.wh_ratio.append(ratio)
                    self.wh_ratio_idx_map.append((lmdb_idx, idx))
                except:
                    continue

        # Convert to numpy array for easier processing
        self.wh_ratio = np.array(self.wh_ratio)
        self.wh_ratio_sort = np.argsort(self.wh_ratio)

    def resize_norm_img(self, data, imgW, imgH, padding=True):
        """
        Resize and normalize image according to target dimensions
        """
        img = data["image"]
        h = img.shape[0]
        w = img.shape[1]

        if not padding:
            resized_image = cv2.resize(
                img, (imgW, imgH), interpolation=cv2.INTER_LINEAR
            )
            resized_w = imgW
        else:
            ratio = w / float(h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(math.ceil(imgH * ratio))
            resized_image = cv2.resize(img, (resized_w, imgH))

        resized_image = resized_image.astype("float32")

        # Normalize image
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5

        # Create padding
        padding_im = np.zeros((3, imgH, imgW), dtype=np.float32)
        padding_im[:, :, :resized_w] = resized_image
        valid_ratio = min(1.0, float(resized_w / imgW))

        data["image"] = padding_im
        data["valid_ratio"] = valid_ratio
        return data

    def __getitem__(self, properties):
        """
        Get an item with specific width, height
        properties is a tuple containing (width, height, index, wh_ratio)
        """
        img_height = properties[1]
        idx = properties[2]

        # Determine image width based on properties
        if self.ds_width and len(properties) > 3 and properties[3] is not None:
            wh_ratio = properties[3]
            # Calculate width based on height and ratio
            img_width = img_height * (
                1 if int(round(wh_ratio)) == 0 else int(round(wh_ratio))
            )

            # Get the file index from sorted ratios
            sorted_idx = self.wh_ratio_sort[idx]
            lmdb_idx, file_idx = self.wh_ratio_idx_map[sorted_idx]
        else:
            # Use regular index ordering
            lmdb_idx, file_idx = self.data_idx_order_list[idx]
            img_width = properties[0]
            wh_ratio = None

        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)

        try:
            # Get sample from LMDB
            sample_info = self.get_lmdb_sample_info(
                self.lmdb_sets[lmdb_idx]["txn"], file_idx
            )

            if sample_info is None:
                raise Exception(f"Sample {file_idx} in LMDB {lmdb_idx} does not exist!")

            img_buf, label = sample_info

            data = {"image": img_buf, "label": label}

            # Get external data if needed
            data["ext_data"] = self.get_ext_data()
            
            if self.read_masks:
                data["masks"] = self.get_masks_data(self.lmdb_sets[lmdb_idx]["txn"], file_idx)

            # Apply transformations except the last one
            outs = transform(data, self.ops[:-1])

            if outs is not None:
                # Apply resize and normalization
                outs = self.resize_norm_img(outs, img_width, img_height)
                # Apply final transformation
                outs = transform(outs, self.ops[-1:])

        except Exception:
            self.logger.error(
                f"Error processing LMDB index {lmdb_idx}, file {file_idx}: {traceback.format_exc()}"
            )
            outs = None

        if outs is None:
            # During evaluation, we should fix the idx to get same results
            # for many times of evaluation
            rnd_idx = (idx + 1) % self.__len__()
            return self.__getitem__([img_width, img_height, rnd_idx, wh_ratio])

        return outs

    def __len__(self):
        if self.ds_width:
            return len(self.wh_ratio)
        else:
            return super().__len__()


class LMDBDataSetSR(LMDBDataSet):
    def buf2PIL(self, txn, key, type="RGB"):
        imgbuf = txn.get(key)
        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        im = Image.open(buf).convert(type)
        return im

    def str_filt(self, str_, voc_type):
        alpha_dict = {
            "digit": string.digits,
            "lower": string.digits + string.ascii_lowercase,
            "upper": string.digits + string.ascii_letters,
            "all": string.digits + string.ascii_letters + string.punctuation,
        }
        if voc_type == "lower":
            str_ = str_.lower()
        for char in str_:
            if char not in alpha_dict[voc_type]:
                str_ = str_.replace(char, "")
        return str_

    def get_lmdb_sample_info(self, txn, index):
        self.voc_type = "upper"
        self.max_len = 100
        self.test = False
        label_key = b"label-%09d" % index
        word = str(txn.get(label_key).decode())
        img_HR_key = b"image_hr-%09d" % index  # 128*32
        img_lr_key = b"image_lr-%09d" % index  # 64*16
        try:
            img_HR = self.buf2PIL(txn, img_HR_key, "RGB")
            img_lr = self.buf2PIL(txn, img_lr_key, "RGB")
        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = self.str_filt(word, self.voc_type)
        return img_HR, img_lr, label_str

    def __getitem__(self, idx):
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        sample_info = self.get_lmdb_sample_info(
            self.lmdb_sets[lmdb_idx]["txn"], file_idx
        )
        if sample_info is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        img_HR, img_lr, label_str = sample_info
        data = {"image_hr": img_HR, "image_lr": img_lr, "label": label_str}
        outs = transform(data, self.ops)
        if outs is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        return outs


class LMDBDataSetTableMaster(LMDBDataSet):
    def load_hierarchical_lmdb_dataset(self, data_dir):
        lmdb_sets = {}
        dataset_idx = 0
        env = lmdb.open(
            data_dir,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        txn = env.begin(write=False)
        num_samples = int(pickle.loads(txn.get(b"__len__")))
        lmdb_sets[dataset_idx] = {
            "dirpath": data_dir,
            "env": env,
            "txn": txn,
            "num_samples": num_samples,
        }
        return lmdb_sets

    def get_img_data(self, value):
        """get_img_data"""
        if not value:
            return None
        imgdata = np.frombuffer(value, dtype="uint8")
        if imgdata is None:
            return None
        imgori = cv2.imdecode(imgdata, 1)
        if imgori is None:
            return None
        return imgori

    def get_lmdb_sample_info(self, txn, index):
        def convert_bbox(bbox_str_list):
            bbox_list = []
            for bbox_str in bbox_str_list:
                bbox_list.append(int(bbox_str))
            return bbox_list

        try:
            data = pickle.loads(txn.get(str(index).encode("utf8")))
        except:
            return None

        # img_name, img, info_lines
        file_name = data[0]
        bytes = data[1]
        info_lines = data[2]  # raw data from TableMASTER annotation file.
        # parse info_lines
        raw_data = info_lines.strip().split("\n")
        _raw_name, text = (
            raw_data[0],
            raw_data[1],
        )  # don't filter the samples's length over max_seq_len.
        text = text.split(",")
        bbox_str_list = raw_data[2:]
        bbox_split = ","
        bboxes = [
            {"bbox": convert_bbox(bsl.strip().split(bbox_split)), "tokens": ["1", "2"]}
            for bsl in bbox_str_list
        ]

        # advance parse bbox
        # import pdb;pdb.set_trace()

        line_info = {}
        line_info["file_name"] = file_name
        line_info["structure"] = text
        line_info["cells"] = bboxes
        line_info["image"] = bytes
        return line_info

    def __getitem__(self, idx):
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        data = self.get_lmdb_sample_info(self.lmdb_sets[lmdb_idx]["txn"], file_idx)
        if data is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        outs = transform(data, self.ops)
        if outs is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        return outs

    def __len__(self):
        return self.data_idx_order_list.shape[0]
