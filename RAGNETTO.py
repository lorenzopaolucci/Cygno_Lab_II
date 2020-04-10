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
sigma = 10          # Sigma per la generazione del blob

eps_range = []       
best_min_samples = []
best_eff = []

# Genero L x L pixel

grid = np.zeros((L,L))        # Contiene le "coordinate" di ogni pixel, e il valore del segnale
background = np.zeros((L,L))  # Griglia background
signal = np.zeros((L,L))      # Griglia segnale

n_bkg = 0                     # Contatore per il background
n_signal = 0                  # Contatore per il segnale         

for i in np.arange(0,L,1):    # Riempio i pixel di un background uniforme

  for j in np.arange(0,L,1):

    r = np.random.normal(0,1)
    background[i][j] = r
    n_bkg+=1

# Genero il blob

X, blob_labels = make_blobs(n_samples=n_samples, centers=centers, cluster_std=sigma, random_state=0)

# Discretizzazione del blob

for j in np.arange(0,len(X),1):

  r = abs(np.random.normal(1,0.5))

  x = int(np.round(X[j][0]))
  y = int(np.round(X[j][1]))

  signal[x][y] = r
  n_signal+=1

# Somma background+segnale

for j in np.arange(0,L,1):

  for i in np.arange(0,L,1):

    s = signal[i][j] + background[i][j]

    grid[i][j] = np.round(s)

# Creazione array coordinate per DBSCAN

points_list= []

for i in np.arange(0,L,1):

  for j in np.arange(0,L,1):

      x = j
      y = i
      z = grid[i][j]
      points_list.append([x,y,z])
      
points = np.array(points_list)
