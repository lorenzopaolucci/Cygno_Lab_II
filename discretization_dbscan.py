import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


L = 100             # Dimensione griglia
th = 1              # Soglia
centers= [[50,50]]  # Centro del blob
n_samples = 1000    # Numero di segnali generati
sigma = 10          # Sigma per la generazione del blob


# Genero L x L pixel secondo una gaussiana mu=0, sigma=1

pixel = np.zeros((L,L,1))  # Contiene le "coordinate" di ogni pixel, e il valore del segnale 

background = 0             # Contatore per il fondo

for i in np.arange(0,L,1): # Riempio i pixel di un background uniforme

  for k in np.arange(0,L,1):

    bkg = np.random.normal(0,1) 

    if bkg < th:           # Solo se sopra soglia il segnale viene considerato

      pixel[i][k][0] = 0

    else:

      pixel[i][k][0] = 1

      background+=1

# Genero il blob

X, blob_labels = make_blobs(n_samples=n_samples, centers=centers, cluster_std=sigma, random_state=0)

# Discretizzazione del blob

X_discrete = np.zeros_like(X)

for i in np.arange(0,2,1):

  for k in np.arange(0,len(X),1):

    X_discrete[k][i] = np.around(X[k][i]) 

# Riempio la griglia

signal = 0                      # Contatore di quanto segnale "nuovo" viene aggiunto sul fondo

for k in np.arange(0,len(X),1):

  x = int(X_discrete[k][0])     # np.around ha come output lo stesso tipo dell'input, quindi float 
  y = int(X_discrete[k][1])

  if pixel[x][y][0] == 0:

    pixel[x][y][0] = 1
    
    signal+=1

# Plot di segnale+rumore

for i in np.arange(0,L,1):    

  for k in np.arange(0,L,1):

    if pixel[i][k][0] != 0:

      plt.plot(k,i,'.', markerfacecolor='r', markeredgecolor='r', markersize=5)

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10, 10)
plt.show()

print('Background points: %d\nNew signal over background (out of %d generated): %d' %(background,n_samples,signal))
