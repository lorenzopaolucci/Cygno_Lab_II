import numpy as np
import matplotlib
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

#PARAMETRI#
n_pixel = 100       #numero di pixel per ogni coordinata => viene generata una griglia n_pixel x n_pixel
n_signal = 100      #numero di punti generati dalla gaussiana
gaus_center_x = 49  #ascissa del centro della gaussiana del segnale
gaus_center_y = 49  #ordinata del centro della gaussiana del segnale
gaus_sigma = 10     #sigma della gaussiana del segnale
noise_sigma = 0.2   #sigma della generazione del rumore

#FUNZIONI PER LA GENERAZIONE DI SEGNALE E RUMORE#
#signal genera n_points punti centrati in (x, y) con sigma=sigma, controlla se in quel punto la griglia è vuota e, in questo caso, cambia il valore da 0 a 1
def signal(grid, n_points, x, y, sigma):
  for i in range(0, n_points, 1):
    control = True
    while control==True:  #vengono generate le coordinate (x,y) del segnale e si verifica che la griglia sia precedentemente vuota nel punto
      x_point = x + int(np.random.normal(0, sigma))
      y_point = y + int(np.random.normal(0, sigma))

      if (int(grid[x_point][y_point]) == 0): 
        control = False
        grid[x_point][y_point] = 1

  return grid

#noise per ogni punto della griglia stabilisce (con gaussiana centrata in zero e sigma=sigma) se mettere il rumore -1 oppure no
def noise(grid, n, sigma):
  counts = 0 #conta quanti punti di rumore ci sono
  for i in range(0, n):
    for j in range(0, n):
      control = True
      while control==True:
        if abs(np.random.normal(0, sigma))>0.5: 
          if grid[i][j]==0:
            control = False
            grid[i][j] = -1
            counts += 1
        else: control = False

  return grid, counts

grid = np.zeros((n_pixel, n_pixel))
grid = signal(grid, n_signal, gaus_center_x, gaus_center_y, gaus_sigma)
grid, counts = noise(grid, n_pixel, noise_sigma)


#CLUSTERIZZAZIONE CON DBSCAN
eps = 1
min_samples = 10

db = DBSCAN(eps, min_samples).fit(grid)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool) # Inizializza un array booleano, della stessa forma di labels_
core_samples_mask[db.core_sample_indices_] = True         # Considera tutti i core trovati da dbscan
labels = db.labels_  

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) # Conta i cluster, togliendo il  rumore (k=-1)
n_noise_ = list(labels).count(-1)                           # Numero di punti di rumore

#if info_cluster != 0:
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# Plot di segnale+rumore

for i in range(0,n_pixel,1):    
  for k in range(0,n_pixel,1):

    if grid[i][k] != 0 and grid[i][k] != -1:
      plt.plot(k,i,'.', markerfacecolor='r', markeredgecolor='r', markersize=5)
    
    elif grid[i][k] == -1:
      plt.plot(k,i,'.', markerfacecolor='b', markeredgecolor='b', markersize=5)

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10, 10)
plt.show()
