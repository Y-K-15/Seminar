## 必要なライブラリのインポート:
# numpy（数値計算用ライブラリ）、
# pandas（データ操作用ライブラリ）、
# scipy.stats（統計解析用ライブラリ）、
# matplotlib.pyplot（グラフ描画用ライブラリ）
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

## パラメータの定義:
# Define the number of trials and observations
num_trials = 100
num_obs = 1000
# Define alpha, beta, and sigma
alpha = 0
beta = 2
sigma = 400

# Initialize a list to store beta_hat values
# beta_hat（βの推定値）を格納するための空のリスト
beta_hat_values = []


# Run trials
# 試行のループ
for j in range(num_trials):## 0~99

    # Generate random component u ~ N(0, sigma)
    # ランダムコンポーネントの生成
    u = np.random.normal(0, sigma, num_obs) 


    # Create x vector
    # 1からnum_obsまでの整数を要素に持つxベクトルを生成
    x = np.arange(1, num_obs+1)

    # Generate y #y_i 
    y = alpha + beta * x + u

    # Evaluate the log-likelihood for each potential beta_hat
    max_log_likelihood = -np.inf
    beta_hat = 0
    # 1から3まで0.01刻みの値をbeta_hat(ここではi)として試す
    # それぞれの値での対数尤度を計算
    # 課題7~9の処理
    # 等差数列を生成 #np.arange(1, 3.01, 0.01)1~3までの0.01刻みのリスト
    # iはbeta hatになりうる数値
    for beta_tilda in np.arange(1, 3.01, 0.01):
    # for beta_tilda in np.arange(1, 3.01, 0.001):
        u_tilda = y - (alpha + beta_tilda * x)
        pdf = norm.pdf(u_tilda, 0, sigma) # yの確率密度関数を求める。u_iについてpdfを求めれば良い
        log_likelihood = np.sum(np.log(pdf)) # log_likelihoodは確立密度関数の対数の和
        if log_likelihood > max_log_likelihood: #max_log_likelihoodと比較して、それより大きかったら代入する。
            max_log_likelihood = log_likelihood
            beta_hat = beta_tilda # log_likelihoodを最大にするベータの値を保存する。
        # print(max_log_likelihood) # max_log_likelihoodを出力

    # Append the beta_hat value
    # それぞれのループのbeta hatの値を配列に入れる。
    beta_hat_values.append(beta_hat)

#ループの終わり



# Convert list to a DataFrame
#  beta_hat_valuesリストをpandasのデータフレームに変換
df = pd.DataFrame(beta_hat_values, columns=['beta_hat'])

# Plot histogram of beta_hat values
# ヒストグラムに表示する
plt.hist(df['beta_hat'], bins=100, edgecolor='k') # pandasのデータ増えrーむdfから'beta_hat'を取り出して、グラフのためのデータとする。'bins=10'でヒストグラムの棒の数を10本にする。
plt.xlabel('Beta Hat') #x軸ラベル設定
plt.ylabel('Frequency') #y軸ラベル設定
plt.title('Histogram of Beta Hat') #グラフのタイトル設定
plt.show() #表示する
