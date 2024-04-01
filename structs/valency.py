import numpy as np
from typing import List
from structs.dataset import DataSet


class Valency:
    """Represents the valency information of a molecule.

    Attributes:
      va: Valency of bond type a.
      vb: Valency of bond type b.
      vc: Valency of bond type c.
      va_full: Full valency of bond type a.
      vb_full: Full falency of bond type b.
      vc_full: Valency of bond type c.

    Two valency objects are equal if their attributes are all equal.
    """

    def __init__(self, G, max_per_group, num_bb):
        self.G = G
        self.max_per_group = max_per_group
        self.num_bb = num_bb

    def __eq__(self, obj):
        return (self.total == obj.total and self.va == obj.va and self.vb == obj.vb and self.vc == obj.vc)

    @property
    def total(self):
        return self.val_to_arr("valency")

    @property
    def va(self):
        return self.val_to_arr("va")

    @property
    def vb(self):
        return self.val_to_arr("vb")

    @property
    def vc(self):
        return self.val_to_arr("vc")


    def va_full(self, bb):
        ds = DataSet.instance()
        return np.array([ds.valences[1][i] for i in bb]).astype(bool)


    def vb_full(self,bb):
        ds = DataSet.instance()
        return np.array([ds.valences[2][i] for i in bb]).astype(bool)


    def vc_full(self,bb):
        ds = DataSet.instance()
        return np.array([ds.valences[3][i] for i in bb]).astype(bool)

    def tobytes(self):
        return self.va.tobytes() + self.vb.tobytes() + self.vc.tobytes()

    def val_to_arr(self, feat):
        arr_1d = self.val_to_arr_1d(feat)
        return np.reshape(arr_1d, (self.num_bb, self.max_per_group))

    def val_to_arr_1d(self, feat):
        val_vec = np.zeros(self.max_per_group * self.num_bb)
        if self.G.number_of_nodes() == 0:
            return val_vec
        bb_idx = np.array([v["bb_idx"] for _, v in self.G.nodes(data=True)])
        inner_idx = np.array([v["inner_idx"]
                             for _, v in self.G.nodes(data=True)])
        feat = np.array([v[feat] for _, v in self.G.nodes(data=True)])
        idxs = []
        for bb, inner in zip(bb_idx, inner_idx):
            idxs.append(bb * self.max_per_group + inner)

        idxs = np.array(idxs)
        val_vec[idxs] = feat
        return val_vec

    @property
    def get_v(self, type):
        if type == 0:
            return self._va
        if type == 1:
            return self._vb
        if type == 2:
            return self._vc

    def __repr__(self):
        msg = (
            f"{'Total:': <7}{self.total}\n"
            f"{'va:': <7}{self.total}\n"
            f"{'vb:': <7}{self.total}\n"
            f"{'vc:': <7}{self.total}\n"
        )
        return msg
