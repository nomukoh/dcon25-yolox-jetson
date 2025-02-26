#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        """データセットの場所を定義"""
        self.data_dir = "./datasets/dataset"
        self.train_ann = "train_annotations.json"
        self.val_ann = "val_annotations.json"

        """クラス数の変更"""
        self.num_classes = 8

        """評価間隔を変更（初期設定では 10epoch ごとに評価）"""
        self.eval_interval = 1
