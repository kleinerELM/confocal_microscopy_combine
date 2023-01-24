import os, time, cv2, math
import numpy as np
import matplotlib.pyplot as plt
# check for dependencies
home_dir = os.path.dirname(os.path.realpath(__file__))


### actual program start
if __name__ == '__main__':
    file_before = home_dir + os.sep + 'C3S-Balken z-Daten_vorher.asc'
    file_after  = home_dir + os.sep + 'C3S-Balken z-Daten_nachher.asc'

    print( "loading C3S-Balken z-Daten_vorher.asc" )
    before = np.loadtxt(file_before, delimiter="\t", dtype=str )
    before = before[0:before.shape[0],0:before.shape[1]-1].astype(np.float64)
    print( np.min(before), np.max(before), np.mean(before) )

    plt.hist(before, bins=1000)  # arguments are passed to np.histogram

    plt.title("Vorher")
    plt.savefig( home_dir + os.sep + 'hist_vorher.png' )
    plt.show()

    print( "loading C3S-Balken z-Daten_nachher.asc" )
    after  = np.loadtxt(file_after,  delimiter="\t", dtype=str )
    after = after[0:after.shape[0],0:after.shape[1]-1].astype(np.float64)
    print( np.min(after),  np.max(after),  np.mean(after)  )

    plt.hist(after, bins=1000)  # arguments are passed to np.histogram

    plt.title("Nacher")
    plt.savefig( home_dir + os.sep + 'hist_vorher.png' )
    plt.show()

    #plt.savefig( home_dir + os.sep + 'hist_nachher.png' )

    print( "Script DONE!" )