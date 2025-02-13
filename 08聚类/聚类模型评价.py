# 导入必要的模块
from sklearn import metrics
from sklearn import datasets
import pandas as pd
# 导入聚类库
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, SpectralClustering

data = datasets.load_digits()

X, y = data.data, data.target
# 算法
algorithms = []
# k均值
algorithms.append(KMeans(n_clusters=10, random_state=1))
# 近邻传播
algorithms.append(AffinityPropagation())
# 谱聚类
algorithms.append(SpectralClustering(n_clusters=10, random_state=1,
                                     affinity='nearest_neighbors'))
# 凝聚聚类
algorithms.append(AgglomerativeClustering(n_clusters=10))

data = []
for algo in algorithms:
    algo.fit(X)
    data.append(({
        'ARI': metrics.adjusted_rand_score(y, algo.labels_),
        'AMI': metrics.adjusted_mutual_info_score(y, algo.labels_, average_method='arithmetic'),
        'Homogenity': metrics.homogeneity_score(y, algo.labels_),
        'Completeness': metrics.completeness_score(y, algo.labels_),
        'V-measure': metrics.v_measure_score(y, algo.labels_),
        'Silhouette': metrics.silhouette_score(X, algo.labels_)}))

results = pd.DataFrame(data=data, columns=['ARI', 'AMI', 'Homogenity', 'Completeness',
                                           'V-measure', 'Silhouette'],
                       index=['K-means', 'Affinity', 'Spectral', 'Agglomerative'])
print(results)
