from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import axes3d
import numpy as np


info_cluster=0                #if !=0 print numero di cluster individuati, numero di punti rumore, parametri l'efficienza
plot_cluster=0                #if !=0 plot dei cluster individuati
print_eff=0                  #if !=0 print efficienza per ogni run
plot_purity=0                #if !=0 plot segnale individuato da dbscan rispetto segnale 'vero'

eps_range = []
min_samples_range = []
efficiency_best = []
purity_plot= []
min_samples_plot = []
eps_plot=[]

purity_highest = []
efficiency_from_purity = []

min_samples_purity = []
eps_purity = []

min_eps = 4
max_eps = 8
step_eps = 0.5

max_purity=0
max_efficiency=0


for eps in np.arange(min_eps, max_eps, step_eps):

    min_min_samples = 2*eps
    max_min_samples = 10*eps
    step_min_samples = eps

    purity_colored=[] 
    min_samples_colored=[] 

    found=[]
    
    for min_samples in np.arange(min_min_samples, max_min_samples, step_min_samples):

        # CLUSTERING

        eps_range.append(eps)
        min_samples_range.append(min_samples)

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

        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
        for each in np.linspace(0, 1, len(unique_labels))]             # Sceglie la palette di   colori senza il nero

        efficiency = 0
        weight_sum = 0

        clusters_points = (labels==-1)  #seleziono tutti i punti clusterizzati tranne il rumore
       
        
        punti=points[~clusters_points] #punti clusterizzati da dbscan come segnale e basta
       

        for k, col in zip(unique_labels, colors):                      # Per ogni cluster, associo un colore
        
            class_member_mask = (labels == k)                          # Seleziona tutti i punti del cluster k

            xy_core = points[class_member_mask & core_samples_mask]    # Solo se è nel cluster E è un core point
            xy_border = points[class_member_mask & ~core_samples_mask] # Solo se è nel cluster E non è core  ==  è un edge point del cluster
            
            # Efficienza della clusterizzazione

            phot = 0
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
              eff_partial = ((tot_signal - abs(tot_signal-len(xy_core)-len(xy_border)) )/tot_signal)*1/dist

              if eff_partial < 0:
                
                efficiency += 0

              else:

                efficiency += eff_partial
              
              weight_sum += 1/dist

              

            if plot_cluster != 0:

              plt.plot(xy_core[:, 0], xy_core[:, 1], '.',markerfacecolor=tuple(col),
                    markeredgecolor=tuple(col), markersize=5)

              plt.plot(xy_border[:, 0], xy_border[:, 1], '.',markerfacecolor=tuple(col),
                    markeredgecolor=tuple(col), markersize=5)

        #purezza

        if len(punti) > 0: 

          len_sig=len(signal_plot)
          len_punti=len(punti)

          count_p=0
          for i in range (len_sig):
            for j in range (len_punti):
              if punti[j,0]==signal_plot[i,0] and punti[j,1]==signal_plot[i,1] and punti[j,2]==signal_plot[i,2]:
                count_p+=1
        
            
          purity=1-(len(punti)-count_p)/count_p


          #variabili grafico colorato
          purity_colored.append(purity)
          min_samples_colored.append(min_samples)
          Y=np.array(purity_colored)
          X=np.array(min_samples_colored)
          
          #variabili grafico 3d
          purity_plot.append(purity)
          min_samples_plot.append(min_samples)
          eps_plot.append(eps)
 
          z=np.array(purity_plot)
          Z=np.expand_dims(z,axis=0)
          y=np.array(min_samples_plot)
          x=np.array(eps_plot)

          plt.plot(X,Y)
           
          
          
          if plot_purity!=0:

            print('punti segnale dbscan=%d   punti segnale vero=%d   intersezione=%d  purity=%f   eps=%.3f   min_samples=%.3f' %(len_punti,len_sig,count_p,purity,eps,min_samples)) 
            plt.title('plot del segnale trovato da dbscan vs segnale vero')
            plt.plot(punti[:,0],punti[:,1],'r.')
            plt.plot(signal_plot[:,0],signal_plot[:,1],'b.')
            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(6,6)
            plt.show()

                
         
        if n_clusters_ != 0:

          efficiency /= weight_sum
          
          efficiency_best.append(efficiency)

        else:
          
          efficiency_best.append(0)
          
        if print_eff != 0:
          print('Eps: %.2f Min_samples: %.2f Efficiency: %f Purity: %f Efficiency*Purity: %f' %(eps,min_samples,efficiency,purity,efficiency*purity))


        if purity==1:
          purity_highest.append(purity)
          efficiency_from_purity.append(efficiency)
          min_samples_purity.append(min_samples)
          eps_purity.append(eps)
          combined=np.column_stack((purity_highest,efficiency_from_purity,min_samples_purity,eps_purity))
          

        if efficiency > max_efficiency:
          minsample_best_eff=min_samples
          eps_best_eff=eps
          max_efficiency=efficiency
          purity_efficiency=purity


        if plot_cluster != 0:

          plt.title('Eps=%.1lf, min_samples=%d, estimated number of clusters: %d' % (eps,min_samples,n_clusters_))
          fig = matplotlib.pyplot.gcf()
          fig.set_size_inches(6,6)
          plt.show()

print('Efficiency max: %f  with eps=%.2f  min_samples=%.2f  relative purity: %f ' %(max_efficiency,eps_best_eff,minsample_best_eff,purity_efficiency))
plt.show()

#print('Threshold: %.2f Eps: %.2f Min_samples: %.2f Best efficiency: %f Best Purity: %f Overall efficiency: %f' %(th,eps_range[index],min_samples_range[index],efficiency_best[index],purity_best[index],max_efficiency))

fig = plt.figure()
ax = fig.gca(projection='3d')
surf=ax.plot_trisurf(x,y,z,cmap='plasma')
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10, 10)
fig.colorbar(surf)
plt.show()
