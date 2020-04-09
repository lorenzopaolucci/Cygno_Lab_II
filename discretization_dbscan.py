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
plot_3d=0      #if !=0 plot istogramma eps vs min samples vs eff
print_eff=0    #if !=0 print efficienza per ogni run


L = 100             # Dimensione griglia
centers = [[49,49]] # Centro del blob
n_samples = 1000    # Numero di segnali generati
sigma = 10          # Sigma per la generazione del blob


for th in np.arange(0.5,1.5,0.1):

    eps_range = []       
    best_min_samples = []
    best_eff = []

    # Genero L x L pixel secondo una gaussiana mu=0, sigma=1

    grid = np.zeros((L,L))     # Contiene le "coordinate" di ogni pixel, e il valore del segnale

    background = 0             # Contatore per il fondo

    for i in np.arange(0,L,1): # Riempio i pixel di un background uniforme

      for j in np.arange(0,L,1):

        bkg = np.random.normal(0,1)

        if bkg < th:           # Solo se sopra soglia il segnale viene considerato

          grid[i][j] = 0

        else:

          grid[i][j] = 1

          background+=1

    # Genero il blob

    X, blob_labels = make_blobs(n_samples=n_samples, centers=centers, cluster_std=sigma, random_state=0)

    # Discretizzazione del blob

    X_discrete = np.zeros_like(X)

    for i in np.arange(0,2,1):

      for j in np.arange(0,len(X),1):

        X_discrete[j][i] = np.around(X[j][i])

    # Riempio la griglia

    signal = 0                      # Contatore di quanto segnale "nuovo" viene aggiunto sul fondo

    for j in np.arange(0,len(X),1):

      x = int(X_discrete[j][0])     # np.around ha come output lo stesso tipo dell'input, quindi float
      y = int(X_discrete[j][1])

      if grid[x][y] == 0:

        grid[x][y] = 1
        
        signal+=1

    # Plot di segnale+rumore & creazione array coordinate

    points_list= []

    for i in np.arange(0,L,1):

      for j in np.arange(0,L,1):

        if grid[i][j] != 0:

          plt.plot(j,i,'.', markeredgecolor='r', markersize=5)

          x = j
          y = i
          points_list.append([x,y])
          
    points = np.array(points_list)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 10)
    plt.show()
    print('Background points: %d\nNew signal over background (out of %d generated): %d' %(background,n_samples,signal))


    # STUDIO EFFICIENZA DI DBSCAN


    ##############################################################################


    for eps in np.arange(1, 10, 0.5):

        # plot
        efficiency_noise_list = []
        min_samples_range = []


        for min_samples in np.arange(2, 16, 1):

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
            # Efficienza
            if n_noise_ <= expected_noise:
              eff_noise = 1 - (expected_noise-n_noise_)/(expected_noise)
              
            else:
              eff_noise = 1- (n_noise_-expected_noise)/(n_samples)

            efficiency_noise_list.append(eff_noise)
            eff_cluster = 0
            weight = 0
      
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
                  plt.plot(xy_core[:, 0], xy_core[:, 1], '.',
                        markeredgecolor=tuple(col), markersize=5)

                  plt.plot(xy_border[:, 0], xy_border[:, 1], '.',
                        markeredgecolor=tuple(col), markersize=5)
                  
            if plot_cluster != 0:
              plt.title('Eps=%.1lf, min_samples=%d, estimated number of clusters: %d' % (eps,min_samples,n_clusters_))
              fig = matplotlib.pyplot.gcf()
              fig.set_size_inches(10, 10)
              plt.show()

            if print_eff != 0:
              print('Noise efficiency: %.3lf \tCluster efficiency: %.3lf\n' %eff_noise)


        if plot_2d != 0:
          plt.figure(1)
          plt.title('Andamento Efficiency Noise in funzione di min_samples con eps=%lf' % eps)
          plt.plot(min_samples_range,efficiency_noise_list, 'ro',min_samples_range,efficiency_noise_list,'k')
          plt.xlabel('Min samples')
          plt.ylabel('Efficiency Noise')
          plt.xlim(-0.1, max(min_samples_range)+5)
          plt.show()

        eps_range.append(eps)
        best_min_samples.append(min_samples_range[efficiency_noise_list.index(max(efficiency_noise_list))])
        best_eff.append(max(efficiency_noise_list))


        if info_cluster !=0 or plot_cluster !=0 or plot_2d !=0 or print_eff !=0:
          print('\n ################################################################## \n')

    index = best_eff.index(max(best_eff))
    eps = eps_range[index]
    min_samples = best_min_samples[index]

    print(eps,min_samples,max(best_eff))

    db = DBSCAN(eps, min_samples).fit(points)                   
    core_samples_mask = np.zeros_like(db.labels_,dtype=bool)    
    core_samples_mask[db.core_sample_indices_] = True           
    labels = db.labels_
    unique_labels = set(labels)

    hist = np.zeros(20)
    xy = []

    for k in unique_labels:

      if k != -1:

        class_member_mask = (labels == k)
        xy_core = points[class_member_mask & core_samples_mask]
        xy_border = points[class_member_mask & ~core_samples_mask]

        xy = np.concatenate((xy_core,xy_border))

        for j in np.arange(0,len(xy),1):
          
          x = xy[j][0]

          hist[int(x/5)] += 1

    range_x = range(0,len(hist),1)

    plt.bar(range_x,hist)

    plt.show()
