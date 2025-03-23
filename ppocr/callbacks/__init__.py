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
from __future__ import unicode_literals

__all__ = ["build_callbacks"]

import copy

from paddle.callbacks import EarlyStopping
from paddle.hapi.callbacks import CallbackList


def build_callbacks(callbacks_list, model):
    support_dict = ["EarlyStopping"]

    cbs = []
    for callback in callbacks_list:
        assert isinstance(callback, dict) and len(callback) == 1, "yaml format error"
        cb_name = list(callback)[0]
        assert cb_name in support_dict
        param = {} if callback[cb_name] is None else callback[cb_name]
        cb = eval(cb_name)(**param)
        cbs.append(cb)
    callbacks = CallbackList(cbs)
    callbacks.set_model(model)
    return callbacks
