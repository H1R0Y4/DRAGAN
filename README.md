# DRAGAN
論文の[リンク](https://arxiv.org/abs/1705.07215)です。
## Usage


```
$ python SNDRAGAN.py --img_path <img_path> --out_dir <output_dir_path>
```
必要な設定
- --img_path  :<img_path>には使いたい画像を保管しているディレクトリを指定する。(.pngまで指定する必要はない)
- --out_dir   :<output_dir_path> は結果の出力ディレクトリの指定をする。

任意の設定
- --size      :Generatorの100次元から生成する際のサイズを設定する。64x64の場合は4、128x128のときは8を指定する。デフォルトは4。
- --n_epochs  :エポック数の設定をする。デフォルトは1000。10000epoch以上にした際、コードの書き換えが必要。
- --interval  :何枚置きに画像を保存するか設定をする。デフォルトは10。
- --batch_size:バッチサイズを設定する。　デフォルトは100。
## Result
自分の環境で生成した画像を以下に示す。  
![image](https://github.com/H1R0Y4/DRAGAN/blob/master/0900.png)
- 900 epoch目
- バッチサイズは100
- 最適化手法はAdam(α=0.0001, β1=0.5)
- Gradient Penalty項の重みλは10.0、Cは10.0
