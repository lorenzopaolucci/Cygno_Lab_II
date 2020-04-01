import numpy as np
import matplotlib
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


info_cluster=0 #if !=0 print numero di cluster individuati, numero di punti rumore, parametri l'efficienza
plot_cluster=0 #if !=0 plot dei cluster individuati
plot_2d=0      #if !=0 plot di eff vs min_samples per ogni eps
plot_3d=1      #if !=0 plot istogramma eps vs min samples vs eff
print_eff=0    #if !=0 print efficienza per ogni run


##############################################################################
# Genera punti nel piano, gaussiani centrati in centers
centers = [[1, 0], [-1, 0]]
n_samples = 1000
sigma = 0.4
X, labels_true = make_blobs(n_samples=n_samples, centers=centers, cluster_std=sigma,
                            random_state=0)

#X = StandardScaler().fit_transform(X) #rinormalizza media=0, std=1

# plot_3d
eps_range = [] 
min_samples_range = []
efficiency_list = []

##############################################################################


for eps in np.arange(0.1, 0.4, 0.05):

    #plot_2d
    b=[]
    c=[]

    for min_samples in np.arange(2, 20, 1):

        if info_cluster !=0 or plot_cluster !=0 or plot_2d !=0 or print_eff !=0:
          print('############         Eps: %f Min_Samples: %f         ############\n' %(eps,min_samples))
      

        #plot_3d
        eps_range.append(eps)
        min_samples_range.append(min_samples)


        db = DBSCAN(eps, min_samples).fit(X)                      # CLUSTERING
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool) # Inizializza un array booleano, della stessa forma di labels_
        core_samples_mask[db.core_sample_indices_] = True         # Considera tutti i core trovati da dbscan
        labels = db.labels_
        
        
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) # Conta i cluster, togliendo il  rumore (k=-1)
        n_noise_ = list(labels).count(-1)                           # Numero di punti di rumore

        if info_cluster != 0:
          print('Estimated number of clusters: %d' % n_clusters_)
          print('Estimated number of noise points: %d' % n_noise_)

        ##############################################################################
        # Efficienza
        
        eff_noise = 1-n_noise_/n_samples
        eff_cluster = 0 
        weight = 0 
        eff_noise_weight = 0.5
        eff_cluster_weight = 0.5                            

        ##############################################################################
        # Plot
        import matplotlib.pyplot as plt

        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
        for each in np.linspace(0, 1, len(unique_labels))] # Sceglie la palette di   colori senza il nero

        for k, col in zip(unique_labels, colors):   # Per ogni cluster, associo un colore
            if k == -1:
              col = [0, 0, 0, 1]                    # Nero per il rumore
        
            class_member_mask = (labels == k)       # Seleziona tutti i punti del cluster k

            xy_core = X[class_member_mask & core_samples_mask]    # Solo se è nel cluster E è un core point
            xy_border = X[class_member_mask & ~core_samples_mask] # Solo se è nel cluster E non è core  ==  è un edge point del cluster
            
            if plot_cluster != 0:
              plt.plot(xy_core[:, 0], xy_core[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6)

              plt.plot(xy_border[:, 0], xy_border[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6)
            
            if k != -1 and n_clusters_ >= len(centers): # Solo se # cluster > # centri generati
              x = 0 
              y = 0
              for ic in xy_core:
                x += ic[0]
                y += ic[1]
              for ib in xy_border:
                x += ib[0]
                y += ib[1]
              
              n_members=list(labels).count(k)           # Membri cluster k 

              distance = []
              distance_from_centers = []

              for i_centers in centers:
                distance.append( np.sqrt( ((y/(n_members)-i_centers[1])**2
                                                   + ((x/(n_members))-i_centers[0])**2 ))) # Dist media dei membri dai diversi centri generati

              distance_from_centers.append(min(distance))                                  # Centro più vicino

              expected = (n_samples-n_noise_)/len(centers)                                 # Num atteso di membri (meno il rumore)
              
              occurrence = (expected - abs(expected - (len(xy_border)+len(xy_core)))) / (expected) # Differenza rispetto all'atteso

              eff_cluster += occurrence / distance_from_centers[-1]                                # Eff_cluster: somma delle differenze pesate con 1/dist dal centro più vicino
              
              weight += 1/distance_from_centers[-1]
              
              if info_cluster !=0:
                print('\n\tcluster label %.lf'%(k))
                print('distanza del cluster da uno dei centri: %lf' %(distance_from_centers[-1]))
                print('occurences: %lf' %occurrence)
                print('elementi nel cluster: %lf' %(len(xy_border)+len(xy_core)))
                print('elementi in tutti i cluster: %lf' %(n_samples-n_noise_))
                print('somma numeratore efficienza cluster: %lf' %(eff_cluster))
                print('pesi: %lf' %(weight))
              
        if plot_cluster != 0:
          plt.title('Eps=%.1lf, min_samples=%d, estimated number of clusters: %d' % (eps,min_samples,n_clusters_))
          plt.show()
        
        if  n_clusters_ >= len(centers):
          eff_cluster /= weight
        
        efficiency = eff_noise_weight*eff_noise + eff_cluster_weight*eff_cluster            # Efficienza totale: media eff_noise eff_cluster

        efficiency_list.append(eff_noise_weight*eff_noise + eff_cluster_weight*eff_cluster) # plot_3d

        # plot_2d
        b.append(efficiency)
        c.append(eff_cluster)

        if print_eff != 0:
          print('Noise efficiency: %.3lf \tCluster efficiency: %.3lf\n' %(eff_noise, eff_cluster))
          print('Total efficiency: %.3lf\n\n' %efficiency)


    if plot_2d != 0:
      plt.figure(1)
      plt.title('Andamento Efficiency in funzione di min_samples con eps=%lf' % eps)
      plt.plot(min_samples_range,b, 'ro',min_samples_range,b,'k')
      plt.xlabel('Min samples')
      plt.ylabel('Efficiency')
      plt.xlim(-0.1, max(min_samples_range)+5)
      plt.ylim(-0.1, 1.1)
    
    

      #plt.figure(2) #Terry
      #plt.title('Andamento Efficiency in funzione di min_samples con eps=%lf' % eps)
      #plt.plot(a,c,'g^',a,c,'b')
      #plt.xlabel('Min samples')
      #plt.ylabel('Efficiency Cluster')
      #plt.xlim(-0.1, max(a)+5)
      #plt.ylim(-0.1, 1.1)
      plt.show()

    if info_cluster !=0 or plot_cluster !=0 or plot_2d !=0 or print_eff !=0:
      print('\n ################################################################## \n')

if plot_3d != 0:
  fig = plt.figure()
  dx = np.full_like(eps_range, 0.04)
  dy = np.full_like(min_samples_range, 1)
  z = np.zeros_like(efficiency_list)
  ax = fig.add_subplot(111, projection='3d')
  plt.title('Efficiency Histogram', fontsize=20) 
  plt.xlabel('eps', fontsize=15)
  plt.ylabel('min samples', fontsize=15)
  fig.set_size_inches(11,8) 
  ax.bar3d(eps_range, min_samples_range, z, dx, dy, efficiency_list, color='g')
  plt.show()

max_index = efficiency_list.index(max(efficiency_list))

print('The maximum value of efficiency is: %.3lf \tfor eps: %.2lf, min_samples: %d' %(max(efficiency_list), eps_range[max_index], min_samples_range[max_index]))
