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
    x = np.linspace(0,len(hist),1000)

    plt.bar(range_x,hist/(sum(hist)*5))
    plt.plot(x, np.exp(-0.5*(x-49/5)**2/((sigma/5)**2) )/(2*np.pi*sigma), color='yellow', linewidth=2)

    plt.show()
  
