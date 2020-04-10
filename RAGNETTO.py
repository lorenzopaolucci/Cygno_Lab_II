import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

info_cluster=0 #if !=0 print numero di cluster individuati, numero di punti rumore, parametri l'efficienza
plot_cluster=0 #if !=0 plot dei cluster individuati
plot_2d=0      #if !=0 plot di eff vs min_samples per ogni eps
print_eff=0    #if !=0 print efficienza per ogni run

L = 100             # Dimensione griglia
centers = [[49,49]] # Centro del blob
n_samples = 1000    # Numero di segnali generati
sigma = 1.5         # Sigma per la generazione del blob
sigma_spatial = 10  # Sigma della distribuzione spaziale del blob


for th in np.arange(1,4,1):

    eps_range = []       
    best_min_samples = []
    best_eff = []

    # Genero L x L pixel secondo una gaussiana mu=0, sigma=1

    grid = np.zeros((L,L))     # Contiene le "coordinate" di ogni pixel, e il valore del segnale
    background = np.zeros((L,L))  # Griglia background
    signal = np.zeros((L,L))      # Griglia segnale

    for i in np.arange(0,L,1):    # Riempio i pixel di un background uniforme

      for j in np.arange(0,L,1):

        r = np.random.normal(0,2)
        background[i][j] = r

    # Genero il blob

    X, blob_labels = make_blobs(n_samples=n_samples, centers=centers, cluster_std=sigma_spatial, random_state=0)

    # Discretizzazione del blob

    for j in np.arange(0,len(X),1):

      r = abs(np.random.normal(10,sigma))
      x = int(np.round(X[j][0]))
      y = int(np.round(X[j][1]))

      signal[x][y] = r

    # Somma background+segnale
    count=0
    for j in np.arange(0,L,1):

      for i in np.arange(0,L,1):

        s = signal[i][j] + background[i][j]
        if s >= th:
          grid[i][j] = np.round(s)
          count += 1
      
    print(count)
    # Creazione array coordinate per DBSCAN

    points_list= []

    for i in np.arange(0,L,1):

      for j in np.arange(0,L,1):

        if grid[i][j] != 0:

          points_list.append([j,i,grid[i][j]])

    points = np.array(points_list)

    plt.scatter(points[:,0],points[:,1],c=points[:,2],cmap='gist_heat')
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(15, 15)
    plt.colorbar()
    plt.show()

    ##############################################################################
    # STUDIO EFFICIENZA DI DBSCAN

    for eps in np.arange(1, 6, 1):

        # plot
        efficiency_noise_list = []
        min_samples_range = []


        for min_samples in np.arange(2, 8, 2):

            min_samples_range.append(min_samples)

            db = DBSCAN(eps, min_samples).fit(points)                   # CLUSTERING
            core_samples_mask = np.zeros_like(db.labels_,dtype=bool)    # Inizializza un array booleano, della stessa forma di labels_
            core_samples_mask[db.core_sample_indices_] = True           # Considera tutti i core trovati da dbscan
            labels = db.labels_
            
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) # Conta i cluster, togliendo il  rumore (k=-1)
            n_noise_ = list(labels).count(-1)                           # Numero di punti di rumore
            expected_noise = signal+background-n_samples

            if info_cluster != 0:
              print('Estimated number of clusters: %d' % n_clusters_)
              print('Estimated number of noise points: %d' % n_noise_)

            ##############################################################################
            import matplotlib.pyplot as plt

            unique_labels = set(labels)
            colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))] # Sceglie la palette di   colori senza il nero

            for k, col in zip(unique_labels, colors):          # Per ogni cluster, associo un colore
                if k == -1:
                  col = [0, 0, 0, 1]                           # Nero per il rumore
            
                class_member_mask = (labels == k)              # Seleziona tutti i punti del cluster k

                xy_core = points[class_member_mask & core_samples_mask]    # Solo se è nel cluster E è un core point
                xy_border = points[class_member_mask & ~core_samples_mask] # Solo se è nel cluster E non è core  ==  è un edge point del cluster
                
                if plot_cluster != 0:
                  plt.plot(xy_core[:, 0], xy_core[:, 1], '.',markerfacecolor=tuple(col),
                        markeredgecolor=tuple(col), markersize=5)

                  plt.plot(xy_border[:, 0], xy_border[:, 1], '.',markerfacecolor=tuple(col),
                        markeredgecolor=tuple(col), markersize=5)
                  
            if plot_cluster != 0:
              plt.title('Eps=%.1lf, min_samples=%d, estimated number of clusters: %d' % (eps,min_samples,n_clusters_))
              fig = matplotlib.pyplot.gcf()
              fig.set_size_inches(10, 10)
              plt.show()

