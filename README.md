# OthelloAI

本稿は、東京大学松尾研究室主催の「深層強化学習」講座の最終課題のテーマとして、オセロAIの作成を目指すものです。オセロは、状態集合や行動集合が有限であり、初学者にとっても比較的アルゴリズムが理解しやすく、またその戦略が複雑であるため、強化学習の教材として適していると、個人的に考えています。

## オセロの環境構築

オセロ公式サイト(https://www.megahouse.co.jp/othello/what/)によると、オセロのルールは以下の通りです。
1. 2人のプレイヤーは、白い石を打つ側と、黒い石を打つ側に分かれます。はじめに、盤の中央に、図のように石を並べます。この配置から始めるのが、オセロの決まりです。黒が先手になります。
2. 石を打つときは、自分の色の石で相手の色の石をはさめるマスに打ちます。たて・よこ・ななめ、いずれの方向にはさんでもかまいません。相手の石をはさめないマスに石を打つことはできません。はさんだ石は必ずすべてひっくり返して、自分の色の石にします。
3. 盤の四隅の石は、はさむことはできません。
4. 相手の石をはさめるマスがない場合は、パスとなります。打てるマスができるまでは、何回でもパスとなり、相手が連続して打ちます。
5. 盤面がぜんぶ埋まるか、黒白ともに打つマスがなくなったら、終局です。その時点で自分の色の石が多い方が勝ちとなります。

## 実装方針

1. オセロの状態を表すクラスを作成する。
   1. 正常に動作していることを確認するために、人対人のオセロを実装する。
2. オセロの状態を受け取り、次の行動の確率分布を返す方策ニューラルネットワークをクラスとして作成する。
   1. 盤面の状態を入力とし、各マスに石を置く確率を出力する。
3. オセロの状態を受け取り、その状態価値を返す価値ニューラルネットワークをクラスとして作成する。
   1. 盤面の状態を入力とし、その状態の価値(勝利の見込み)を出力する。
4. 自己対戦を行い、経験を蓄積する、プレイヤークラスを作成する。
   1. 自己対戦では、行動の先の状態の価値と、方策ネットワークの出力を踏まえた、AlphaZero指標が最も高くなる手を選択する。
5. 実際に、自己対戦を行い、学習を行う。
6. 学習したモデルを保存できるようにし、それを用いて、人間と対戦する。

## 実行方法

上から順にセルを実行していくことで、オセロAIの作成を行うことができます。
その途中途中で、実装したクラスの動作確認を行うことが可能です。
- 人間vs人間のオセロ
- AIの自己対戦
を行うことができます。

一番最後のセルを実行することで、何種類かの学習済みモデルと対戦することができます。

## モデルと性能(自分との対戦結果)

1. (2000ゲーム×2730ループで学習したモデル) policy.pth, Q_value.pth
   ネットワーク構造は、以下の通りです。
   ```
   ＃方策ネットワーク
    self.features = nn.Sequential(
         nn.Conv2d(1, 4, kernel_size=5, padding=2, padding_mode='replicate'),    # out_channelsは欲しい特徴マップの数
         nn.ReLU(),
         nn.Conv2d(4, 16, kernel_size=5, padding=2, padding_mode='replicate'),
         nn.ReLU(),
         nn.Conv2d(16, 16, kernel_size=3, padding=1, padding_mode='replicate'),
         nn.ReLU(),
         nn.Conv2d(16, 4, kernel_size=3, padding=1, padding_mode='replicate'),
         nn.ReLU(),
         nn.Conv2d(4, 1, kernel_size=1)
     )
     self.classifier = nn.Sequential(
         nn.Linear(64, 64),              # 全結合層
         nn.ReLU(),
         nn.Linear(64, 64),
         nn.Softmax(dim=1)               # ミニバッチの各データセットごとに確率分布に変換
     )
   #価値ネットワーク
   self.features = nn.Sequential(
         nn.Conv2d(1, 4, kernel_size=5, padding=2, padding_mode='replicate'),
         nn.ReLU(),
         nn.Conv2d(4, 16, kernel_size=5, padding=2, padding_mode='replicate'),
         nn.ReLU(),
         nn.Conv2d(16, 4, kernel_size=3, padding=1, padding_mode='replicate'),
         nn.ReLU(),
         nn.Conv2d(4, 1, kernel_size=1)
     )
     self.value = nn.Sequential(
         nn.Linear(64, 32),
         nn.ReLU(),
         nn.Linear(32, 16),
         nn.ReLU(),
         nn.Linear(16, 1),
         nn.Tanh()
     )
   ```
2. (2000ゲーム×2730ループで学習したモデルその2) policy2.pth, Q_value2.pth
   ネットワーク構造を以下のように変更。
3. (未定)