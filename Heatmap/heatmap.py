def corrmap(data, cmap="YlGnBu", annot = True):
    
    # Packages
    import seaborn as sns
    import numpy as np
    
    # Main
    corr = np.round(data.corr(),2)
    
    plt.figure(figsize = (6,4), dpi = 120)
    sns.heatmap(corr, cmap=cmap, annot=annot)
    plt.show