info_cluster = 0                #if !=0 print numero di cluster individuati, numero di punti rumore, parametri l'efficienza
plot_cluster = 0                #if !=0 plot dei cluster individuati
print_eff    = 0                #if !=0 print efficienza per ogni run
plot_purity  = 0                #if !=0 plot segnale individuato da dbscan rispetto segnale 'vero'

    detection_efficiency_scatter = []
    purity_scatter = []
    n_clusters = []

    min_eps = 4
    max_eps = 8
    step_eps = 0.5

    max_purity=0
    max_efficiency=0

    for eps in np.arange(min_eps, max_eps, step_eps):

        min_min_samples = 2*eps
        max_min_samples = 10*eps
        step_min_samples = 1

        purity_plot= []
        min_samples_range=[]
        cluster_efficiency_plot = []
        detection_efficiency_plot = []
        min_samples_plot = []
        
        for min_samples in np.arange(min_min_samples, max_min_samples, step_min_samples):

            min_samples_plot.append(min_samples)

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

            # Plot dei cluster individuati

            unique_labels = set(labels)
            colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]             # Sceglie la palette di   colori senza il nero

            cluster_efficiency = 0
            weight_sum = 0

            clusters_points = (labels==-1)  # Seleziona tutti i punti clusterizzati tranne il rumore
            
            punti = points[~clusters_points]  # Punti clusterizzati da dbscan come segnale

            df = pd.DataFrame(data=punti,index=None)
            df.to_csv('data_%.1lf_%.1f.csv' %(eps,min_samples),index_label=False)
           
            for k, col in zip(unique_labels, colors):                      # Per ogni cluster, associo un colore
            
                class_member_mask = (labels == k)                          # Seleziona tutti i punti del cluster k

                xy_core = points[class_member_mask & core_samples_mask]    # Solo se è nel cluster E è un core point
                xy_border = points[class_member_mask & ~core_samples_mask] # Solo se è nel cluster E non è core  ==  è un edge point del cluster
                
                # Efficienza della clusterizzazione

                phot = 0                                      # Contatore di fotoni
                x = 0
                y = 0

                if k == -1:

                  col = [0, 0, 0, 1]                          # Nero per il rumore

                else:

                  for i in np.arange(0,len(xy_core),1):       # Somme sui pixel contenuti nel cluster k, pesate con il numero di fotoni
                    
                    x += xy_core[i][0] * xy_core[i][2]
                    y += xy_core[i][1] * xy_core[i][2]
                    phot += xy_core[i][2]

                  for i in np.arange(0,len(xy_border),1):
                    
                    x += xy_border[i][0] * xy_border[i][2]
                    y += xy_border[i][1] * xy_border[i][2]
                    phot += xy_border[i][2]

                  x /= phot
                  y /= phot

                  dist = np.sqrt((x-centers[0][0])**2+(y-centers[0][1])**2)
                  clust_eff_partial = ((tot_signal - abs(tot_signal-len(xy_core)-len(xy_border)) )/tot_signal)*1/dist

                  if clust_eff_partial < 0:
                    
                    cluster_efficiency += 0

                  else:

                    cluster_efficiency += clust_eff_partial
                  
                  weight_sum += 1/dist


                if plot_cluster != 0:

                  plt.plot(xy_core[:, 0], xy_core[:, 1], '.',markerfacecolor=tuple(col),
                        markeredgecolor=tuple(col), markersize=5)

                  plt.plot(xy_border[:, 0], xy_border[:, 1], '.',markerfacecolor=tuple(col),
                        markeredgecolor=tuple(col), markersize=5)

            # Purezza

            if len(punti) > 0:                                # Considero solo le run che trovano segnale

              len_sig=len(signal_plot)
              len_punti=len(punti)

              count_p=0                                       # Conta i pixel trovati da dbscan correttamente
              for i in range (len_sig):
                for j in range (len_punti):
                  if punti[j,0]==signal_plot[i,0] and punti[j,1]==signal_plot[i,1] and punti[j,2]==signal_plot[i,2]:
                    count_p+=1
            
                
              purity=1-(len(punti)-count_p)/count_p
              detection_efficiency=count_p/len(signal_plot)

              n_clusters.append(n_clusters_)
              detection_efficiency_scatter.append(detection_efficiency)
              detection_efficiency_plot.append(detection_efficiency)
              min_samples_range.append(min_samples)
              purity_plot.append(purity)
              purity_scatter.append(purity)

              # Variabili grafico colorato

              #Y=np.array(purity_colored)
              #X=np.array(min_samples_range)
              
              if plot_purity!=0:

                print('Punti segnale dbscan=%d   Punti segnale vero=%d   Intersezione=%d  purity=%f   eps=%.3f   min_samples=%.3f' %(len_punti,len_sig,count_p,purity,eps,min_samples))
                plt.title('Plot del segnale trovato da dbscan vs segnale vero')
                plt.plot(punti[:,0],punti[:,1],'r.')
                plt.plot(signal_plot[:,0],signal_plot[:,1],'b.')
                fig = matplotlib.pyplot.gcf()
                fig.set_size_inches(6,6)
                plt.show()

            # Efficienza
             
            if n_clusters_ != 0:

              cluster_efficiency /= weight_sum
              
              cluster_efficiency_plot.append(cluster_efficiency)

            else:
              
              cluster_efficiency_plot.append(0)
              
              
            if print_eff != 0:
              print('Eps: %.2f Min_samples: %.2f Efficiency: %f Purity: : %f' %(eps,min_samples,efficiency,purity))

            if cluster_efficiency > max_efficiency:           # Definisce l'efficienza più alta
              minsample_best_eff = min_samples
              eps_best_eff = eps
              max_efficiency = cluster_efficiency
              clust_eff = cluster_efficiency
              det_eff = detection_efficiency
              purity_eff = purity
              n_clust = n_clusters[-1]

            if plot_cluster != 0:

              plt.title('Eps=%.1lf, min_samples=%d, estimated number of clusters: %d' % (eps,min_samples,n_clusters_))
              fig = matplotlib.pyplot.gcf()
              fig.set_size_inches(6,6)
              plt.show()

        # Plot 2d

        plt.subplot(2,2,1)
        plt.title('Efficiency trend as a function of MinSamples',fontsize=18)
        plt.xlabel('Min samples',fontsize=18)
        plt.ylabel('Efficiency',fontsize=18)
        plt.plot(min_samples_plot,cluster_efficiency_plot,'.-',label='Eps = %.1lf' %eps)

        plt.subplot(2,2,2)
        plt.title('Purity trend as a function of MinSamples',fontsize=18)
        plt.xlabel('Min samples',fontsize=18)
        plt.ylabel('Purity',fontsize=18)
        plt.plot(min_samples_range,purity_plot,'.-',label='Eps = %.1lf' %eps)

        plt.subplot(2,2,3)
        plt.title('Detection efficiency trend as a function of MinSamples',fontsize=18)
        plt.xlabel('Min samples',fontsize=18)
        plt.ylabel('Efficiency',fontsize=18)
        plt.plot(min_samples_range,detection_efficiency_plot,'.-',label='Eps = %.1lf' %eps)

        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(20, 15)

     
    plt.legend(loc='upper right',fontsize=15)

    fig.savefig('plot_purity_efficiency.png')

    print('Cluster Efficiency: %f  with eps=%.2f  min_samples=%.2f  relative purity: %f Detection efficiency: %f Clusters found: %d' %(clust_eff,eps_best_eff,minsample_best_eff,purity_eff,det_eff,n_clust))
    plt.show()

    # Scatter Purity vs Detection Efficiency

    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    plt.scatter(detection_efficiency_scatter,purity_scatter,c=n_clusters,cmap='jet')
    #fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 8)
    ax = plt.colorbar()
    ax.set_label('$N_{Cluster}$',fontsize=18)
    plt.title('Purity vs Detection efficiency and $N_{Cluster}$',fontsize=18)
    plt.xlabel('Detection efficiency',fontsize=18)
    plt.ylabel('Purity',fontsize=18)
    plt.show()
