# 自動車認識 YOLOX-s (JETSON ORIN NANO 用)

本プログラムは、 DCON2025 で自動車の認識を YOLOX-s で行うために使用したものです。

## 著作権表示および引用元

このリポジトリの一部のコードは、[GitHubリポジトリ](https://github.com/Megvii-BaseDetection/YOLOX)から引用しており、また、[zennの記事](https://zenn.dev/opamp/articles/d3878b189ea256)を参考に開発されています。  
引用元の著作権およびライセンス情報は各所有者に帰属します。  
このリポジトリは Apache License 2.0 の条件の下で配布されています。

## 親子関係

本リポジトリは、[親リポジトリ](https://github.com/nomukoh/dcon25-yolox-win)内の YOLOX_jetson フォルダと連携しており、双方でコードや変更が同期されています。

## 開発環境

本リポジトリは Jetson Orin Nano 向けに開発されています。  
Jetson Orin Nano の Linux ベース環境で動作するよう最適化されており、専用の環境設定が施されています。

## 環境構築の手順

Windows 用の手順に従ってください。
・[Github 親リポジトリ](https://github.com/nomukoh/dcon25-yolox-win)

## 学習

```bash
python3 train.py -f yolox_s.py -d 1 -b 16 --fp16 -o -c ./yolox_s.pth --cache
```

## 推論

```bash
python3 demo.py image -f yolox_s.py --device gpu --fp16 --path ./datasets/dataset/inference2017/[検査画像までのパス] -c ./YOLOX_outputs/yolox_s/best_ckpt.pth --save_result
```
