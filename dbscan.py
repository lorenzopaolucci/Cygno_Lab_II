print(__doc__)

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


# #############################################################################
# Genera punti nel piano, gaussiani centrati nelle 4 liste
centers = [[1, 1], [-1, -1], [1, -1],[-2,1.5]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)

X = StandardScaler().fit_transform(X) #rinormalizza media=0, std=1

# #############################################################################
# Parametri DBSCAN 
for eps in np.arange(0.1, 0.4, 0.1)
    for min_samples in np.arange(5,20,5)
    
        db = DBSCAN(eps, min_samples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool) #inizializza un array booleano,   della stessa forma di labels_
        core_samples_mask[db.core_sample_indices_] = True #considera tutti i core trovati da dbscan
        labels = db.labels_
        
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) #conta i cluster, togliendo il  rumore (k=-1)
        n_noise_ = list(labels).count(-1) #numero di punti di rumore
        
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        
        # #############################################################################
        # Plot
        import matplotlib.pyplot as plt
        
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                    for each in np.linspace(0, 1, len(unique_labels))] #sceglie la palette di   colori senza il nero
        for k, col in zip(unique_labels, colors): #per ogni cluster, associo un colore
            if k == -1:
                # Nero per il rumore
                col = [0, 0, 0, 1]
        
            class_member_mask = (labels == k) #seleziona tutti i punti del cluster k
            #plt.style.use("dark_background")
            xy = X[class_member_mask & core_samples_mask] #plot solo se è nel cluster E è un core   point
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=14)
        
            xy = X[class_member_mask & ~core_samples_mask] #plot solo se è nel cluster E non è core     == è un edge point del cluster
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6)
        
        plt.title('Eps=%d, min_samples=%d, estimated number of clusters: %d' % eps,min_samples,n_clusters_)
        plt.show()
