import numpy as np
import matplotlib

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


# #############################################################################
# Genera punti nel piano, gaussiani centrati in centers
centers = [[1, 0], [-1, 0]]
n_samples = 1000
sigma = 0.4
X, labels_true = make_blobs(n_samples=n_samples, centers=centers, cluster_std=sigma,
                            random_state=0)

#X = StandardScaler().fit_transform(X) #rinormalizza media=0, std=1

# Genera il rumore uniformemente se il parametro generate_noise è diverso da zero
#generate_noise = 0
#if generate_noise != 0:
  
eps_range = []
min_samples_range = []
efficiency = []

# #############################################################################
# Parametri DBSCAN 
for eps in np.arange(0.1, 0.5, 0.05):
    for min_samples in np.arange(6, 24, 2):
        eps_range.append(eps)
        min_samples_range.append(min_samples)
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
        # Study of efficiency
        
        eff_noise = 1-n_noise_/n_samples
        #if(n_clusters_ < len(centers)): eff_cluster.append(0)
        eff_cluster = 0 
        weight = 0 
        eff_noise_weight = 0.5
        eff_cluster_weight = 0.5                            
              
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
            xy_core = X[class_member_mask & core_samples_mask] #plot solo se è nel cluster E è un core   point

            plt.plot(xy_core[:, 0], xy_core[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6)
            print('\tcluster label %.lf'%(k))
            #print(xy_core[:, 1])
            xy_border = X[class_member_mask & ~core_samples_mask] #plot solo se è nel cluster E non è core     == è un edge point del cluster
            plt.plot(xy_border[:, 0], xy_border[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6)
            
            if k != -1 and n_clusters_ >= len(centers):
              x = 0 
              y = 0
              for ic in xy_core:
                x += ic[0]
                y += ic[1]
              for ib in xy_border:
                x += ib[0]
                y += ib[1]
              
              distance = []
              distance_from_centers = []
              for i_centers in centers:
                distance.append( np.sqrt( ((y/(len(xy_border)+len(xy_core)))-i_centers[1])**2
                                                   + ((x/(len(xy_border)+len(xy_core)))-i_centers[0])**2 ) ) 
                #print(distance[-1])
              distance_from_centers.append(min(distance))
              print(distance.index(min(distance)))
              #print(len(xy_border)+len(xy_core))
              #print(x/(len(xy_border)+len(xy_core)))        
              #print(y/(len(xy_border)+len(xy_core)))
              print('distanza del cluster da uno dei centri: %lf' %(distance_from_centers[-1]))
              expected = (n_samples-n_noise_)/len(centers)
              #print(expected)
              occurrence = (expected - abs(expected - (len(xy_border)+len(xy_core)))) / (expected)
              print('occurences: %lf' %occurrence)
              eff_cluster += occurrence / distance_from_centers[-1]
              #eff_cluster += (len(xy_border)+len(xy_core))/(distance_from_centers[-1]*(n_samples-n_noise_))
              weight += 1/distance_from_centers[-1]
              print('elementi nel cluster: %lf' %(len(xy_border)+len(xy_core)))
              print('elementi in tutti i cluster: %lf' %(n_samples-n_noise_))
              print('somma numeratore efficienza cluster: %lf' %(eff_cluster))
              print('pesi: %lf' %(weight))
              
        
        plt.title('Eps=%.1lf, min_samples=%d, estimated number of clusters: %d' % (eps,min_samples,n_clusters_), fontsize=15)

        if  n_clusters_ >= len(centers):
          eff_cluster /= weight
        #aritmetic mean
        efficiency.append(eff_noise_weight*eff_noise + eff_cluster_weight*eff_cluster)
        #geometric mean
        #efficiency.append(np.sqrt(eff_noise*eff_cluster))

        plt.show()

        print('Noise efficiency: %.3lf \tCluster efficiency: %.3lf' %(eff_noise, eff_cluster))
        print('Total efficiency: %.3lf' %efficiency[-1])

fig = plt.figure()
dx = np.full_like(eps_range, 0.04)
dy = np.full_like(min_samples_range, 1)
z = np.zeros_like(efficiency)
ax = fig.add_subplot(111, projection='3d')
plt.title('Efficiency Histogram', fontsize=20) 
plt.xlabel('eps', fontsize=15)
plt.ylabel('min samples', fontsize=15)
fig.set_size_inches(11,8) 
ax.bar3d(eps_range, min_samples_range, z, dx, dy, efficiency, color='g')
plt.show()
max_index = efficiency.index(max(efficiency))
print('The maximum value of efficiency is: %.3lf \tfor eps: %.1lf, min_samples: %d' %(max(efficiency), eps_range[max_index], min_samples_range[max_index]))
