import os, sys, time
import numpy   as np
import sklearn
import sklearn.svm, sklearn.metrics

# -- [HOW TO] -- #
#
# * evaluate__computationalTime( trainer= ( function, trainer ), Data= ( np.array, Data ), \
#                                labels = ( np.array, labels  ) )
#

# ========================================================= #
# ===  evaluate__computationalTime.py                   === #
# ========================================================= #

def evaluate__computationalTime( trainer=None, Data=None, labels=None, trainer_params=None, \
                                 train_size=0.7, shuffle=True, random_state=1, display=True, \
                                 returnType="time" ):

    timeDict    = {}

    # ------------------------------------------------- #
    # --- [1] arguments                             --- #
    # ------------------------------------------------- #
    if ( trainer        is None ): sys.exit( "[evaluate__computationalTime.py] trainer == ???" )
    if ( Data           is None ): sys.exit( "[evaluate__computationalTime.py] Data    == ???" )
    if ( labels         is None ): sys.exit( "[evaluate__computationalTime.py] labels  == ???" )
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
    timeDict["train_start"] = time.perf_counter()
    classifier  = trainer( **Data_labels, **trainer_params )
    timeDict["train_end"]   = time.perf_counter()

    labels_pred = classifier.predict( Data_test )
    accuracy    = sklearn.metrics.accuracy_score( labels_pred, labels_test )  * 100.0

    # ------------------------------------------------- #
    # --- [4] evaluation of computationalTime       --- #
    # ------------------------------------------------- #
    timeDict["pred_start"] = time.perf_counter()
    labels_pred = classifier.predict( Data_test )
    timeDict["pred_end"]   = time.perf_counter()

    # ------------------------------------------------- #
    # --- [5] display indices                       --- #
    # ------------------------------------------------- #
    timeDict["train_duration"] = timeDict["train_end"] - timeDict["train_start"]
    timeDict["pred_duration"]  = timeDict["pred_end"]  - timeDict["pred_start"]
    if ( display ):
        print( "\n" )
        print( "[evaluate__computationalTime.py] <<<< computational Time >>>>" )
        print( "[evaluate__computationalTime.py]  check Accuracy     :: {:.3f} (%)"\
               .format( accuracy ) )
        print()
        print( "[evaluate__computationalTime.py]   << training >>" )
        print( "[evaluate__computationalTime.py] training Size       :: {}" \
               .format( Data_train.shape ) )
        print( "[evaluate__computationalTime.py] train duration      :: {:.6e}" \
               .format( timeDict["train_duration"] ) )
        print( "[evaluate__computationalTime.py] trainingTime / unit :: {:.6e}"\
               .format( timeDict["train_duration"] / labels_train.shape[0] ) )
        
        print()
        print( "[evaluate__computationalTime.py]   << testing  >>" )
        print( "[evaluate__computationalTime.py]     test Size       :: {}"\
               .format( Data_test.shape  ) )
        print( "[evaluate__computationalTime.py] prediction duration :: {:.6e}"\
               .format( timeDict["pred_duration"]  ) )
        print( "[evaluate__computationalTime.py] prediction / unit   :: {:.6e}"\
               .format( timeDict["pred_duration"] / labels_test.shape[0] ) )
        print( "\n" )

    # ------------------------------------------------- #
    # --- [6] return indices                        --- #
    # ------------------------------------------------- #
    if   ( returnType.lower() in [ "list" ] ):
        ret = [ classifier, timeDict ]
    elif ( returnType.lower() in [ "dict" ] ):
        ret = { "classifier":classifier, "timeDict":timeDict }
    elif ( returnType.lower() in [ "time", "timedict" ] ):
        ret = timeDict
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
    x3MinMaxNum = [ 0.0, 0.0,   1 ]
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
    # --- [3] call evaluate__computationalTime      --- #
    # ------------------------------------------------- #
    ret = evaluate__computationalTime( trainer=trainer_for_svm, Data=Data, \
                                       labels=labels, returnType="timeDict" )
    print( ret )
