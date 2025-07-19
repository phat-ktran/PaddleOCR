from typing import List, Optional
import numpy as np
from scipy import sparse
import paddle

import ppocr.data.imaug.utils.equations as eqs


class Connections:
    def __init__(
        self,
        forward,
        backward,
        forward_start,
        forward_end,
        backward_start,
        backward_end,
        is_sparse=False,
    ):
        self.forward = forward
        self.backward = backward
        self.forward_start = forward_start
        self.forward_end = forward_end
        self.backward_start = backward_start
        self.backward_end = backward_end
        self._sparse = is_sparse

    def size(self):
        size = self.forward_start.shape[0]

        if self._sparse:
            size = self.forward_start.shape[1]

        return size

    def is_sparse(self):
        return self._sparse

    def _dense_to_sparse(self, matrix):
        return sparse.csr_matrix(matrix)

    def _sparse_to_dense(self, matrix):
        return matrix.toarray()

    def to_sparse(self):
        if not self._sparse:
            self.forward = self._dense_to_sparse(self.forward)
            self.backward = self._dense_to_sparse(self.backward)
            self.forward_start = self._dense_to_sparse(self.forward_start)
            self.forward_end = self._dense_to_sparse(self.forward_end)
            self.backward_start = self._dense_to_sparse(self.backward_start)
            self.backward_end = self._dense_to_sparse(self.backward_end)
            self._sparse = True

    def to_dense(self):
        if self._sparse:
            self.forward = self._sparse_to_dense(self.forward)
            self.backward = self._sparse_to_dense(self.backward)

            # The .reshape(-1) transforms the shape of the vector from (1, N) to (N,). SciPy does not keep the original
            # vector shape.
            self.forward_start = self._sparse_to_dense(self.forward_start).reshape(-1)
            self.forward_end = self._sparse_to_dense(self.forward_end).reshape(-1)
            self.backward_start = self._sparse_to_dense(self.backward_start).reshape(-1)
            self.backward_end = self._sparse_to_dense(self.backward_end).reshape(-1)
            self._sparse = False

    def extend(self, target_size):
        is_sparse = self._sparse
        if is_sparse:
            self.to_dense()

        current_size = self.size()

        if target_size == current_size:
            extended_connections = Connections(
                np.copy(self.forward),
                np.copy(self.backward),
                np.copy(self.forward_start),
                np.copy(self.forward_end),
                np.copy(self.backward_start),
                np.copy(self.backward_end),
            )

        elif target_size > current_size:
            forward = Connections._extend_2d(self.forward, target_size)
            forward_start = Connections._extend_1d(self.forward_start, target_size)
            forward_end = Connections._extend_1d(self.forward_end, target_size)

            backward = Connections._extend_2d(self.backward, target_size)
            backward_start = Connections._extend_1d(self.backward_start, target_size)
            backward_end = Connections._extend_1d(self.backward_end, target_size)

            extended_connections = Connections(
                forward,
                backward,
                forward_start,
                forward_end,
                backward_start,
                backward_end,
            )
        else:
            extended_connections = None

        if is_sparse:
            self.to_sparse()

            if extended_connections is not None:
                extended_connections.to_sparse()

        return extended_connections

    @staticmethod
    def _extend_2d(matrix, target_size):
        height, width = matrix.shape
        extended_matrix = np.zeros((target_size, target_size), dtype=matrix.dtype)
        extended_matrix[:height, :width] = matrix

        return extended_matrix

    @staticmethod
    def _extend_1d(vector, target_size):
        length = vector.shape[0]
        extended_vector = np.zeros(target_size, dtype=vector.dtype)
        extended_vector[:length] = vector

        return extended_vector

    @staticmethod
    def from_confusion_network(confusion_network, blank_idx=0, dtype=np.float32):
        labeling = eqs.construct_labeling(confusion_network, blank_idx)
        label_probs = [
            eqs.p_symbol(confusion_network, symbol, tau, blank_idx)
            if tau < len(confusion_network)
            else 1.0
            for (symbol, tau) in labeling
        ]
        label_probs = np.array(label_probs, dtype=dtype)

        forward_matrix = np.full((len(labeling), len(labeling)), -1.0, dtype=dtype)

        for i, (symbol1, tau1) in enumerate(labeling):
            for j, (symbol2, tau2) in enumerate(labeling):
                forward_matrix[i, j] = eqs.p_transition(
                    confusion_network, symbol1, tau1, symbol2, tau2, blank_idx
                )

        backward_matrix = np.transpose(forward_matrix)

        forward_init_vector = eqs.alpha_init(
            confusion_network, labeling, blank_idx, dtype=dtype
        )
        backward_init_vector = eqs.beta_init(
            confusion_network, labeling, blank_idx, dtype=dtype
        )
        forward_loss_vector = backward_init_vector
        backward_loss_vector = forward_init_vector / label_probs

        return Connections(
            forward_matrix,
            backward_matrix,
            forward_init_vector,
            forward_loss_vector,
            backward_init_vector,
            backward_loss_vector,
        )


class BatchConnections:
    def __init__(self, forward, backward, forward_start, forward_end, backward_start, backward_end):
        self.forward = forward
        self.backward = backward
        self.forward_start = forward_start
        self.forward_end = forward_end
        self.backward_start = backward_start
        self.backward_end = backward_end

    def to(self, device):
        self.forward = paddle.to_tensor(self.forward, place=device)
        self.forward_start = paddle.to_tensor(self.forward_start, place=device)
        self.forward_end = paddle.to_tensor(self.forward_end, place=device)
        self.backward = paddle.to_tensor(self.backward, place=device)
        self.backward_start = paddle.to_tensor(self.backward_start, place=device)
        self.backward_end = paddle.to_tensor(self.backward_end, place=device)
        return self

    def paddle(self, dtype="float32"):
        self.forward = paddle.to_tensor(self.forward, dtype=dtype)
        self.forward_start = paddle.to_tensor(self.forward_start, dtype=dtype)
        self.forward_end = paddle.to_tensor(self.forward_end, dtype=dtype)
        self.backward = paddle.to_tensor(self.backward, dtype=dtype)
        self.backward_start = paddle.to_tensor(self.backward_start, dtype=dtype)
        self.backward_end = paddle.to_tensor(self.backward_end, dtype=dtype)
        return self

    def numpy(self):
        self.forward = self.forward.numpy()
        self.forward_start = self.forward_start.numpy()
        self.forward_end = self.forward_end.numpy()
        self.backward = self.backward.numpy()
        self.backward_start = self.backward_start.numpy()
        self.backward_end = self.backward_end.numpy()
        return self

    def place(self):
        return self.forward_start.place

    def size(self):
        return self.forward_start.shape[1]

    def __len__(self):
        return self.forward_start.shape[0]

    def __getitem__(self, index):
        return Connections(self.forward[index], self.backward[index],
                           self.forward_start[index], self.forward_end[index],
                           self.backward_start[index], self.backward_end[index])

    def __str__(self):
        output = f"Forward transitions: {self.forward.shape}\n"
        output += f"Forward init vector: {self.forward_start.shape}\n"
        output += f"Forward loss vector: {self.forward_end.shape}\n"
        output += f"Backward transitions: {self.backward.shape}\n"
        output += f"Backward init vector: {self.backward_start.shape}\n"
        output += f"Backward loss vector: {self.backward_end.shape}"
        return output

    @staticmethod
    def stack_connections(connections: List[Connections], target_connections_size: Optional[int] = None):
        sparse_connections = []
        for c in connections:
            if c.is_sparse():
                c.to_dense()
                sparse_connections.append(True)
            else:
                sparse_connections.append(False)

        connections_sizes = [c.size() for c in connections]
        if target_connections_size is not None:
            connections_sizes += [target_connections_size]

        target_connections_size = max(connections_sizes)
        extended_connections = [c.extend(target_connections_size) for c in connections]

        for sparse, c in zip(sparse_connections, connections):
            if sparse:
                c.to_sparse()

        forward = np.stack([c.forward for c in extended_connections])
        forward_start = np.stack([c.forward_start for c in extended_connections])
        forward_end = np.stack([c.forward_end for c in extended_connections])
        backward = np.stack([c.backward for c in extended_connections])
        backward_start = np.stack([c.backward_start for c in extended_connections])
        backward_end = np.stack([c.backward_end for c in extended_connections])

        return BatchConnections(
            forward, backward, forward_start, forward_end, backward_start, backward_end
        )
        

def stack_labels(labels, blank, target_size=None):
    if target_size is None:
        target_size = max([len(l) for l in labels])

    for index, l in enumerate(labels):
        labels_padding = [blank] * (target_size - len(l))
        labels[index] += labels_padding
        labels[index] = np.array(labels[index])

    labels = np.stack(labels)

    return labels


def calculate_target_size(sizes: List[int], size_coefficient=64):
    return int(np.ceil(np.max(sizes) / size_coefficient) * size_coefficient)


def convert_characters_to_labels(confusion_network, character_set):
    new_confusion_network = []

    for confusion_set in confusion_network:
        new_confusion_set = {}
        for character in confusion_set:
            if character is None:
                new_confusion_set[None] = confusion_set[None]
            else:
                new_confusion_set[character_set.index(character)] = confusion_set[
                    character
                ]

        new_confusion_network.append(new_confusion_set)

    return new_confusion_network