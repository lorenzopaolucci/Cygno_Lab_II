import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

info_cluster=0      #if !=0 print numero di cluster individuati, numero di punti rumore, parametri l'efficienza
plot_cluster=1      #if !=0 plot dei cluster individuati
plot_2d=0           #if !=0 plot di eff vs min_samples per ogni eps
print_eff=0         #if !=0 print efficienza per ogni run

L = 100             # Dimensione griglia
centers = [[L*0.5-1,L*0.5-1]] # Centro del blob
n_samples = 10000    # Numero di segnali generati
sigma = 1.5         # Sigma per la generazione del blob
sigma_spatial = 10  # Sigma della distribuzione spaziale del blob

min_th = 1 
max_th = 2
step_th = 1

min_eps = 3
max_eps = 6
step_eps = 0.5

min_min_samples = 2
max_min_samples = 26
step_min_samples = 2


for th in np.arange(min_th,max_th,step_th):

    eps_range = []       
    best_min_samples = []
    best_eff = []

    # Genero L x L pixel 

    grid = np.zeros((L,L))        # Contiene le "coordinate" di ogni pixel, e il valore del segnale
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

      r = abs(np.random.uniform(1,2))
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

        if signal[i][j] == 0 and background[i][j] >= th: # Aggiorno background: contiene pixel di "puro" rumore, cioè pixel con segnale nullo e sopra soglia

          background[i][j] = np.round(background[i][j])

        else:

          background[i][j] = 0 

    # Creazione array coordinate per DBSCAN

    points_list= []

    for i in np.arange(0,L,1):

      for j in np.arange(0,L,1):

        if grid[i][j] != 0:

          points_list.append([j,i,grid[i][j]])

    points = np.array(points_list)

    #plt.scatter(points[:,0],points[:,1],c=points[:,2],cmap='cool')
    #fig = matplotlib.pyplot.gcf()
    #fig.set_size_inches(15, 15)
    #plt.colorbar()
    #plt.show()
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf=ax.plot_trisurf(points[:,0],points[:,1],points[:,2],cmap='plasma')
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 10)
    fig.colorbar(surf)
    plt.show()


    # STUDIO EFFICIENZA DI DBSCAN

    for eps in np.arange(min_eps, max_eps, step_eps):

        # Plot efficienza
        efficiency_noise_list = []
        min_samples_range = []

        for min_samples in np.arange(min_min_samples, max_min_samples, step_min_samples):

            min_samples_range.append(min_samples)

            # CLUSTERING

            db = DBSCAN(eps, min_samples).fit(points)                   
            core_samples_mask = np.zeros_like(db.labels_,dtype=bool)       # Inizializza un array booleano, della stessa forma di labels_
            core_samples_mask[db.core_sample_indices_] = True              # Considera tutti i core trovati da dbscan
            labels = db.labels_

            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)    # Conta i cluster, togliendo il  rumore (k=-1)
            n_noise_ = list(labels).count(-1)                              # Numero di punti di rumore

            if info_cluster != 0:

              print('Estimated number of clusters: %d' % n_clusters_)
              print('Estimated number of noise points: %d' % n_noise_)

            mean_dist = []
            members = []

            # Plot dei cluster individuati

            import matplotlib.pyplot as plt

            unique_labels = set(labels)
            colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]             # Sceglie la palette di   colori senza il nero

            for k, col in zip(unique_labels, colors):                      # Per ogni cluster, associo un colore
            
                class_member_mask = (labels == k)                          # Seleziona tutti i punti del cluster k

                xy_core = points[class_member_mask & core_samples_mask]    # Solo se è nel cluster E è un core point
                xy_border = points[class_member_mask & ~core_samples_mask] # Solo se è nel cluster E non è core  ==  è un edge point del cluster
                
                sum_ = 0
                eff_partial = 0
                x = 0
                y = 0

                if k == -1:

                  col = [0, 0, 0, 1]                                       # Nero per il rumore

                else:

                  for i in np.arange(0,len(xy_core),1):
                    
                    x += xy_core[i][0]
                    y += xy_core[i][1]

                  for i in np.arange(0,len(xy_border),1):
                    
                    x += xy_border[i][0]
                    y += xy_border[i][1]

                  dist = np.sqrt((x/(len(xy_core)+len(xy_border))-centers[0][0])**2+(y/(len(xy_core)+len(xy_border))-centers[0][1])**2)

                  mean_dist.append(dist)
                  members.append(len(xy_core)+len(xy_border))

                if plot_cluster != 0:

                  plt.plot(xy_core[:, 0], xy_core[:, 1], '.',markerfacecolor=tuple(col),
                        markeredgecolor=tuple(col), markersize=5)

                  plt.plot(xy_border[:, 0], xy_border[:, 1], '.',markerfacecolor=tuple(col),
                        markeredgecolor=tuple(col), markersize=5)
            
            # Purezza della clusterizzazione
            
            index_noise = (labels == -1)

            noise = points[index_noise & ~core_samples_mask]

            tot_noise = 0
            found = 0
            tot = 0

            for i in np.arange(0,L,1):

              for j in np.arange(0,L,1):

                if background[i][j] != 0:  
                                                                       # Contatore punti di "puro" rumore
                  tot_noise += 1         

                  for k in np.arange(0,len(noise),1):

                    if j == noise[k,0] and i == noise[k,1] and background[i][j] == noise[k,2]:  # Verifica se il pixel di rumore [j,i] è presente nella lista di rumore individuato da DBSCAN
                      
                      found+=1

                if grid[i][j] != 0:
                  
                  tot+=1                 # Conta il numero totale di pixel accesi

            tot_signal = tot-tot_noise   # Numero totale di pixel accesi dal segnale

            false_negatives = len(noise) - found

            purity = found/tot_noise * (1-false_negatives/tot_noise)
              
            # Efficienza della clusterizzazione

            efficiency = 0
            weight_sum = 0

            if n_clusters_ != 0:
              for i in np.arange(0,len(members),1):

                efficiency += (members[i]/tot_signal*1/mean_dist[i])
                weight_sum += 1/mean_dist[i]

              efficiency /= weight_sum

              print(purity,false_negatives,efficiency,eps,min_samples)

            if plot_cluster != 0:

              plt.title('Eps=%.1lf, min_samples=%d, estimated number of clusters: %d' % (eps,min_samples,n_clusters_))
              fig = matplotlib.pyplot.gcf()
              fig.set_size_inches(10, 10)
              plt.show()
