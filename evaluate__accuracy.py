import os, sys
import numpy   as np
import sklearn
import sklearn.svm, sklearn.metrics

# -- [HOW TO] -- #
#
#   * evaluate__accuracy( trainer= ( function, trainer ), Data= ( np.array, Data ), \
#                         labels = ( np.array, labels  ) )
#
# -- -------- -- #

# ========================================================= #
# ===  evaluate__accuracy.py                            === #
# ========================================================= #

def evaluate__accuracy( trainer=None, Data=None, labels=None, trainer_params=None, \
                        train_size=0.7, shuffle=True, random_state=1, display=True, \
                        returnType="dict" ):

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
    labels_pred = classifier.predict( Data_test )
    confuseMat  = sklearn.metrics.confusion_matrix( labels_test, labels_pred )
    accuracy    = np.trace( confuseMat ) / np.sum( confuseMat )
    if ( confuseMat.shape == (2,2) ):
        TN, FP, FN, TP = np.reshape( confuseMat, (-1,) )
        TPR            = TP / ( TP + FN )
        FPR            = FP / ( FP + TN )

    # ------------------------------------------------- #
    # --- [5] display indices                       --- #
    # ------------------------------------------------- #
    if ( display ):
        print( "[evaluate__accuracy.py] accuracy                     :: {}".format( accuracy ) )
        print( "[evaluate__accuracy.py] TPR  ( Truth Positive Rate ) :: {}".format( TPR      ) )
        print( "[evaluate__accuracy.py] FPR  ( False Positive Rate ) :: {}".format( FPR      ) )
        print()
        print( "[evaluate__accuracy.py] confusion Matrix             :: " )
        print( " Horizontal >> prediction, " )
        print( " Vertical   >> Actual / Truth, " )
        print( "  ( TN = {:10} ),    ( FP = {:10} )".format( confuseMat[0,0], confuseMat[0,1] ) )
        print( "  ( FN = {:10} ),    ( TP = {:10} )".format( confuseMat[1,0], confuseMat[1,1] ) )
        print()
    
    # ------------------------------------------------- #
    # --- [6] return indices                        --- #
    # ------------------------------------------------- #
    if ( returnType.lower() == "list" ):
        ret = [ classifier, accuracy, TPR, FPR ]
    elif ( returnType.lower() == "dict" ):
        ret = { "classifier":classifier, "accuracy":accuracy, "TPR":TPR, "FPR":FPR }
    else:
        print( "[evaluate__accuracy.py] returnType == {} ??? [ERROR] ".format( returnType ) )
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
    def trainer_for_svm( Data=None, labels=None ):
        classifier = sklearn.svm.SVC()
        classifier.fit( Data, labels )
        return( classifier )

    # ------------------------------------------------- #
    # --- [3] call evaluate__accuracy               --- #
    # ------------------------------------------------- #
    ret = evaluate__accuracy( trainer=trainer_for_svm, Data=Data, labels=labels )
    print( ret )
