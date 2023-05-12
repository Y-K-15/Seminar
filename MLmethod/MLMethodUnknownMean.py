import numpy as np
import scipy.stats as stats

# サンプルデータ
samples = np.array([130, 160, 190])
# 標準偏差
sigma = 5

# 探索範囲
mu_candidates = np.arange(130, 200, 10)

# 各muについて対数尤度を計算
log_likelihoods = []
for mu in mu_candidates:
    # 正規分布の確率密度関数を用いて対数尤度を計算
    log_likelihood = np.sum(stats.norm.logpdf(samples, mu, sigma))
    log_likelihoods.append(log_likelihood)

# 最大対数尤度のmuを選択
best_mu = mu_candidates[np.argmax(log_likelihoods)]
print(f"The maximum likelihood estimate for the mean is: {best_mu}")

import matplotlib.pyplot as plt

# 対数尤度をプロット
plt.plot(mu_candidates, log_likelihoods)
plt.scatter(best_mu, max(log_likelihoods), color='red')  # 最大対数尤度を持つmuを赤で表示
plt.xlabel("Mean (mu)")
plt.ylabel("Log-Likelihood")
plt.title("Log-Likelihood of the Samples for Different Means")
plt.grid(True)
plt.show()
