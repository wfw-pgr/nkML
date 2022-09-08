import os, sys
import numpy   as np
import sklearn
import sklearn.svm, sklearn.metrics
import nkUtilities.plot1D         as pl1
import nkUtilities.load__config   as lcf
import nkUtilities.configSettings as cfs

# -- [HOW TO] -- #
#
#   * evaluate__ROC( trainer= ( function, trainer ), Data= ( np.array, Data ), \
#                    labels = ( np.array, labels  ) )
#
# -- -------- -- #

# ========================================================= #
# ===  evaluate__ROC.py                                 === #
# ========================================================= #

def evaluate__ROC( trainer=None, Data=None, labels=None, trainer_params=None, pngFile=None, \
                   train_size=0.7, shuffle=True, random_state=1, display=True, \
                   returnType="ROCs-AUC" ):

    # ------------------------------------------------- #
    # --- [1] arguments                             --- #
    # ------------------------------------------------- #
    if ( trainer        is None ): sys.exit( "[evaluate__accuracy.py] trainer == ???" )
    if ( Data           is None ): sys.exit( "[evaluate__accuracy.py] Data    == ???" )
    if ( labels         is None ): sys.exit( "[evaluate__accuracy.py] labels  == ???" )
    if ( trainer_params is None ): trainer_params = {}

    # ------------------------------------------------- #
    # --- [2] train / test splitting                --- #
    # ------------------------------------------------- #
    import nkML.split__train_test as stt
    Data_train, labels_train, Data_test, labels_test = \
        stt.split__train_test( Data=Data, labels=labels, returnType="list", \
                               train_size=train_size, shuffle=shuffle, \
                               random_state=random_state )
    
    # ------------------------------------------------- #
    # --- [3] train classifier                      --- #
    # ------------------------------------------------- #
    Data_labels = { "Data":Data_train, "labels":labels_train }
    classifier  = trainer( **Data_labels, **trainer_params )

    # ------------------------------------------------- #
    # --- [4] evaluation of accuracy                --- #
    # ------------------------------------------------- #
    labels_pred = classifier.predict          ( Data_test )
    scores_pred = classifier.decision_function( Data_test )

    # ------------------------------------------------- #
    # --- [5] calculate ROC ( FPR, TPR )            --- #
    # ------------------------------------------------- #
    FPR, TPR, threshold = sklearn.metrics.roc_curve    ( labels_test, scores_pred )
    AUC                 = sklearn.metrics.roc_auc_score( labels_test, scores_pred )
    ROCs                = np.concatenate( [FPR[:,None],TPR[:,None],threshold[:,None]], axis=1 )

    # ------------------------------------------------- #
    # --- [6] display                               --- #
    # ------------------------------------------------- #
    if ( display ):
        print()
        print( "[evaluate__ROC.py] << ROC >> " )
        print()
        print( ("   {:14}"*3).format( "FPR", "TPR", "Threshold" ) )
        print( "-"*60 )
        for ik,roc in enumerate(ROCs):
            print( ("   {:13.8e}"*3).format( roc[0], roc[1], roc[2] ) )
        print()
        print( "[evaluate__ROC.py] << AUC >> " )
        print()
        print( "[evaluate__ROC.py] AUC       :: {}".format( AUC       ) )
        print()

    # ------------------------------------------------- #
    # --- [7] output .png                           --- #
    # ------------------------------------------------- #
    if ( pngFile ):
        # -- config settings -- #
        config                   = lcf.load__config()
        config                   = cfs.configSettings( configType="plot.def", config=config )
        config["plt_xAutoRange"] = False
        config["plt_yAutoRange"] = False
        config["plt_xRange"]     = [ -0.1, +1.1 ]
        config["plt_yRange"]     = [ -0.1, +1.1 ]
        config["xMajor_auto"]    = False
        config["yMajor_auto"]    = False
        config["xMajor_ticks"]   = np.linspace( 0.0, 1.0, 6 )
        config["yMajor_ticks"]   = np.linspace( 0.0, 1.0, 6 )
        # --  plot  -- #
        label   = "ROC ( AUC = {:.3f} )".format( AUC )
        fig     = pl1.plot1D( config=config, pngFile=pngFile )
        fig.add__plot( xAxis=FPR, yAxis=TPR, label=label, \
                       linestyle="-", linewidth=2.0, \
                       marker="o", markersize=2.0 )
        fig.add__legend()
        fig.set__axis()
        fig.save__figure()
        print( "[evaluate__ROC.py] pngFile :: {}".format( pngFile ) )
        
    # ------------------------------------------------- #
    # --- [8] return                                --- #
    # ------------------------------------------------- #
    if   ( returnType.lower() in [ "list" ] ):
        ret = [ FPR, TPR, threshold, AUC ]
    elif ( returnType.lower() in [ "dist" ] ):
        ret = { "FPR":FPR, "TPR":TPR, "threshold":threshold, "AUC":AUC }
    elif ( returnType.lower() in [ "rocs-auc" ] ):
        ret = [ ROCs, AUC ]
    elif ( returnType.lower() in [ "rocs" ] ):
        ret = ROCs
    elif ( returnType.lower() in [ "auc"  ] ):
        ret = AUC
    else:
        print( "[evaluate__ROC.py] returnType == {} ??? [ERROR]".format( returnType ) )
        sys.exit()
    return( ret )


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    # ------------------------------------------------- #
    # --- [1] make data / labels                    --- #
    # ------------------------------------------------- #
    x_, y_, z_  = 0, 1, 2
    
    import nkUtilities.equiSpaceGrid as esg
    x1MinMaxNum = [ 0.0, 1.0, 21 ]
    x2MinMaxNum = [ 0.0, 1.0, 21 ]
    x3MinMaxNum = [ 0.0, 0.0,  1 ]
    coord       = esg.equiSpaceGrid( x1MinMaxNum=x1MinMaxNum, x2MinMaxNum=x2MinMaxNum, \
                                     x3MinMaxNum=x3MinMaxNum, returnType = "point" )
    coord[:,z_] = np.sqrt( coord[:,x_]**2 + coord[:,y_]**2 )
    Data        = np.copy( coord )
    labels      = np.where( coord[:,z_] > 0.5, 1, 0 )

    # ------------------------------------------------- #
    # --- [2] define trainer                        --- #
    # ------------------------------------------------- #
    def trainer_for_svm( Data=None, labels=None ):
        classifier = sklearn.svm.SVC()
        classifier.fit( Data, labels )
        return( classifier )

    # ------------------------------------------------- #
    # --- [3] call evaluate__accuracy               --- #
    # ------------------------------------------------- #
    ROCs, AUC  =  evaluate__ROC( trainer=trainer_for_svm, Data=Data, labels=labels, \
                                 returnType="ROCs-AUC", pngFile="roc_curve.png" )
    print()
    print( "ROCs" )
    print( ROCs   )
    print()
    print( "AUC"  )
    print( AUC    )
    print()

