runI          = [1856]
run_ped       = 1748
cimax         = 300
cimin         = 0
dataSelection = 'LAB'
rescale       = 512
nsigma        = 1.5
plot_image    = 0


#piedistallo=riferimento --> come la dark della ccd
try: #scarica i file della run 'run_ped' tramite shell
  fh5 = ("run%d_mean.h5" % (run_ped)) #stringa_nome_run
  cmd = 'wget https://raw.githubusercontent.com/gmazzitelli/cygno/master/data/'+fh5+' -O ./data/'+fh5 #stringa del comando
  os.system(cmd) #esegue il comando in shell
  fh5 = ("run%d_sigma.h5" % (run_ped))
  cmd = 'wget https://raw.githubusercontent.com/gmazzitelli/cygno/master/data/'+fh5+' -O ./data/'+fh5
  os.system(cmd)
except:
  print ("No Pedestal file for run %s on remote repo" % run_ped)

#########################

try:
    fileoutm = ("./data/run%d_mean.h5" % (run_ped))
    m_image = cy.read_image_h5(fileoutm) #cy e' cygnus
    PedOverMax = m_image[m_image > cimax].size
    print ("Pedestal mean: %.2f, sigma: %.2f, over th. (%d) %d" %
       (m_image[m_image<cimax].mean(),
        np.sqrt(m_image[m_image<cimax].var()), cimax,
        (m_image>cimax).sum()))
except:
    print ("No Pedestal file for run %s, run script runs-pedestals.ipynb" % run_ped)
    print ("STOP")

    
try:
    fileouts = ("./data/run%d_sigma.h5" % (run_ped))
    s_image = cy.read_image_h5(fileouts)
    print ("Sigma mean: %.2f, sigma: %.2f, over th. (50) %d" %
   (s_image[s_image<50].mean(),
    np.sqrt(s_image[s_image<50].var()),
    (s_image>50).sum()))
except:
    print ("No Sigma file for run %s, run script runs-pedestals.ipynb" % run_ped)
    print ("STOP")

#########################

th_image   = np.round(m_image + nsigma*s_image) #soglia: sopra il valore del piedistallo
print ("light over Th: %.2f " % (th_image.sum()-m_image.sum()))

    
    
for nRi in range(0,len(runI)): #len[runI]=1
    try:
        print ('Download and open file: '+cy.swift_root_file(dataSelection, runI[nRi]))
        tmp_file = cy.swift_download_file(cy.swift_root_file(dataSelection, runI[nRi])) #scarica la raccolta immagini della run, sono istogrammi 2d di root
        print ('Open file: '+tmp_file)
        f  = ROOT.TFile.Open(tmp_file); #apre il file in root
        print ('Find Keys: '+str(len(f.GetListOfKeys())))
        pic, wfm = cy.root_TH2_name(f) #cosa fa questa funzione?
        max_image = len(pic)
        max_wfm = len(wfm)
        print ("# of Images (TH2) Files: %d " % (max_image))
        print ("# of Waveform (TH2) Files: %d " % (max_wfm))
        nImag=max_image
    except:
        print ("ERROR: No file %d" % (runI[nRi]))
        break

    data_to_save = []
    files = ("./data/dbscan_run%d_cmin_%d_cmax_%d_rescale_%d_nsigma_%.1f_ev_%d_ped_%d.txt" %
                     (runI[nRi], cimin, cimax, rescale, nsigma, max_image, run_ped))
    for iTr in range(0, max_image): #per ogni immagine
        if iTr % 10 == 0: # pach in order overcome the problem of ROOT memory garbage
            print ('RUN: ', runI[nRi], 'Event: ', iTr)
            print (iTr, ' >> Close and re-Open: ', tmp_file)
            f.Close()
            f  = ROOT.TFile.Open(tmp_file);

        image = rtnp.hist2array(f.Get(pic[iTr])).T #converte istogramma root in array numpy, pic contiene le immagini
        #rebin dell'immagine per scendere di dimensione (da 2048x2048 a 512x512), diminuisce la risoluzione
        rebin_image     = cy.rebin(image-m_image, (rescale, rescale))
        rebin_th_image  = cy.rebin((th_image-m_image), (rescale, rescale))

        edges           = (rebin_image > rebin_th_image) & (rebin_image < cimax) #prendi tutti i dati sopra threshold
        points          = np.array(np.nonzero(edges)).T.astype(float)
        scaler          = StandardScaler()
        X_scaled        = scaler.fit_transform(points) #normalizza a media 0 e devstd 1, ogni colonna dell'array singolarmente!

        dbscan          = DBSCAN(eps=0.05, min_samples = 2)
        dbscan.fit(points) #clusterizzazione sui punti selezionati

        clusters = dbscan.fit_predict(X_scaled) #valori di possibili cluster nell'immagine 

        core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool) 
        core_samples_mask[dbscan.core_sample_indices_] = True
        labels = dbscan.labels_

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)


        if plot_image != 0:
          unique_labels = set(labels)
          #colors = [plt.cm.Spectral(each)
          #for each in np.linspace(0, 1, len(unique_labels))] #sceglie la palette di   colori senza il nero
          
          #for k, col in zip(unique_labels, colors): #per ogni cluster, associo un colore
          for k in unique_labels:
            #if k == -1: # Nero per il rumore
              #col = [0, 0, 0, 1]
        
            class_member_mask = (labels == k) #seleziona tutti i punti del cluster k
            #plt.style.use("dark_background")
            if k == -1:

              xy = points[class_member_mask & ~core_samples_mask] #plot del rumore
              plt.plot(xy[:, 0], xy[:, 1], '.', markerfacecolor='k',
                    markeredgecolor='k', markersize=1)
              
            else:
              xy = points[class_member_mask & core_samples_mask] #plot solo se è nel cluster E è un core point
              plt.plot(xy[:, 0], xy[:, 1], '.', markerfacecolor='r',
                    markeredgecolor='r', markersize=1)
        
              xy = points[class_member_mask & ~core_samples_mask] #plot solo se è nel cluster E non è core == è un edge point del cluster
              plt.plot(xy[:, 0], xy[:, 1], '.', markerfacecolor='g',
                    markeredgecolor='g', markersize=1)
        
            #plt.title('Estimated number of clusters: %d' % n_clusters_)

        for ic in range (min(dbscan.labels_), max(dbscan.labels_)): #per ogni cluster individuato: -1 è rumore, 0,1,2,3 sono i cluster
            ph = 0.
            #print ("value: ", iTr, ic, dbscan.labels_[ic], min(dbscan.labels_), max(dbscan.labels_))
            yc = points[:,1][dbscan.labels_==ic] #coordinate del cluster 
            xc = points[:,0][dbscan.labels_==ic]
            ph, dim = cy.cluster_par(yc, xc, rebin_image) #numero di fotoni nel cluster
            width, height, pearson = cy.confidence_ellipse_par(yc,xc) #altezza e larghezza dell'ellisse che racchiude un cluster?
            for j in range(0, dim):
                x=int(xc[j])
                y=int(yc[j])
                #ph += rebin_image[y,x]
                if j == 0:
                    x0start = x
                    y0start = y
            x0end = x
            y0end = y
            data_to_save.append([iTr, ic, dim, ph, ph/dim,
                                 x0start, y0start, x0end, y0end, width, height, pearson]) #file output: nrun,nclus,dimensione cluster,numerodifotoni,densità,varie x e y, altezza, larghezza,pearson=correlazione
        np.savetxt(files, data_to_save, fmt='%.10e', delimiter=" ")
        print ("out file", files)
        #if not cy.rm_file(tmp_file):
         #   print (">> File "+tmp_file+" removed")
    col=['iTr', 'ic', 'dim', 'ph', 'ph/dim', 'x0start', 'y0start', 'x0end', 'y0end', 'width', 'height', 'pearson']
    #pd.set_option('display.max_columns', None)
    #pd.set_option('display.max_rows', None)
    #pd.options.display.max_columns = None
    #pd.options.display.max_rows = None
    df=pd.DataFrame(data=data_to_save, columns=col)
    print(df.head(100))
    plt.show()

####### Dati salvati: #######
#iTr è l'indice dell'immagine che viene studiata (100 immagini)
#ic è l'indice di ogni cluster (-1 per il rumore)
#dim è il numero di pixel che compongono il cluster
#ph è il numero(? perchè decimale?) di fotoni nel cluster considerato
#x0start è l'ascissa del pixel da cui DBSCAN ha iniziato a clusterizzare
#y0start è l'ordinata del pixel da cui DBSCAN ha iniziato a clusterizzare
#x0end è l'ascissa del pixel da cui DBSCAN ha finito di clusterizzare
#y0end è l'ordinata del pixel da cui DBSCAN ha finito di clusterizzare
