import os, sys
import numpy   as np
import sklearn
import sklearn.svm, sklearn.metrics
import nkUtilities.draw__heatmap    as dhm

# -- [HOW TO] -- #
#
#   * gridSearch__hyperParameter( trainer= ( function, trainer ), Data= ( np.array, Data ), \
#                                 labels = ( np.array, labels  ) )
#
# -- -------- -- #

# ========================================================= #
# ===  gridSearch__hyperParameter.py                    === #
# ========================================================= #

def gridSearch__hyperParameter( trainer=None, evaluator=None, Data=None, labels=None, \
                                params1_list=None, params2_list=None, params_name=None,\
                                trainer_params=None, pngFile=None, \
                                train_size=0.7, shuffle=True, random_state=1, \
                                xlabel_format="{:.2e}", ylabel_format="{:.2e}", \
                                returnType="scores" ):

    p1_, p2_ = 0, 1
    
    # ------------------------------------------------- #
    # --- [1] arguments                             --- #
    # ------------------------------------------------- #
    if ( trainer        is None ): sys.exit( "[gridSearch__hyperParameter.py] trainer   == ???" )
    if ( Data           is None ): sys.exit( "[gridSearch__hyperParameter.py] Data      == ???" )
    if ( labels         is None ): sys.exit( "[gridSearch__hyperParameter.py] labels    == ???" )
    if ( params1_list   is None ): sys.exit( "[gridSearch__hyperParameter.py] params1_list=???" )
    if ( params2_list   is None ): sys.exit( "[gridSearch__hyperParameter.py] params2_list=???" )
    if ( params_name    is None ): sys.exit( "[gridSearch__hyperParameter.py] params_name =???" )
    if ( trainer_params is None ): trainer_params = {}
    if ( evaluator      is None ):
        def accuracy_evaluator( pred, true ):
            return( sklearn.metrics.accuracy_score( pred, true ) )
        evaluator = accuracy_evaluator
    params1_list = np.array( params1_list )
    params2_list = np.array( params2_list )
    
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
    scores      = np.zeros( (params1_list.shape[0], params2_list.shape[0]) )
    for i1,p1 in enumerate( params1_list ):
        for i2,p2 in enumerate( params2_list ):
            trainer_params[ params_name[p1_] ] = p1
            trainer_params[ params_name[p2_] ] = p2
            classifier    = trainer( **Data_labels, **trainer_params )
            labels_pred   = classifier.predict( Data_test )
            scores[i1,i2] = evaluator( labels_pred, labels_test )
            print( "parames :: ", trainer_params, ",    scores :: ", scores[i1,i2] )
            
    
    # ------------------------------------------------- #
    # --- [4] output .png                           --- #
    # ------------------------------------------------- #
    if ( pngFile is not None ):
        xlabels  = [ xlabel_format.format( val ) for val in params1_list ]
        ylabels  = [ ylabel_format.format( val ) for val in params2_list ]
        position = [ 0.20, 0.20, 0.86, 0.86 ]
        dhm.draw__heatmap( Data=scores, pngFile=pngFile, \
                           xtitle=params_name[p1_], ytitle=params_name[p2_], \
                           xlabels=xlabels, ylabels=ylabels, position=position )
    
    # ------------------------------------------------- #
    # --- [6] return indices                        --- #
    # ------------------------------------------------- #
    if   ( returnType.lower() == "list" ):
        ret = [ classifier, scores ]
    elif ( returnType.lower() == "dict" ):
        ret = { "classifier":classifier, "scores":scores }
    elif ( returnType.lower() == "scores" ):
        ret = scores
    else:
        print( "[gridSearch__hyperParameter.py] returnType == {} ??? [ERROR] ".format( returnType ) )
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
    x1MinMaxNum = [ 0.0, 1.0, 101 ]
    x2MinMaxNum = [ 0.0, 1.0, 101 ]
    x3MinMaxNum = [ 0.0, 0.0,  1 ]
    coord       = esg.equiSpaceGrid( x1MinMaxNum=x1MinMaxNum, x2MinMaxNum=x2MinMaxNum, \
                                     x3MinMaxNum=x3MinMaxNum, returnType = "point" )
    coord[:,z_] = np.sqrt( coord[:,x_]**2 + coord[:,y_]**2 )
    Data        = np.copy( coord )
    labels      = np.where( coord[:,z_] > 0.5, 1, 0 )

    # ------------------------------------------------- #
    # --- [2] define trainer                        --- #
    # ------------------------------------------------- #
    def trainer_for_svm( Data=None, labels=None, C=None, gamma=None ):
        classifier = sklearn.svm.SVC( C=C, gamma=gamma )
        classifier.fit( Data, labels )
        return( classifier )

    # ------------------------------------------------- #
    # --- [3] call gridSearch__hyperParameter       --- #
    # ------------------------------------------------- #
    params1_list = [ 0.2, 0.5, 1.0, 2.0, 5.0 ]
    params2_list = [ 0.2, 0.5, 1.0, 2.0, 5.0 ]
    params_name  = [ "C", "gamma" ]
    pngFile      = "test/heatmap.png"
    ret = gridSearch__hyperParameter( trainer=trainer_for_svm, Data=Data, labels=labels, \
                                      params1_list=params1_list, params2_list=params2_list, \
                                      params_name=params_name, pngFile=pngFile )
    print( ret )
