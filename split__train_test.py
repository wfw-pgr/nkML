import os, sys
import numpy   as np
import sklearn
import sklearn.model_selection as skms


# ========================================================= #
# ===  split__train_test.py                             === #
# ========================================================= #

def split__train_test( Data=None, labels=None, train_size=0.7, shuffle=True, \
                       random_state=1, flatten=True, returnType="list" ):

    # ------------------------------------------------- #
    # --- [1] arguments                             --- #
    # ------------------------------------------------- #
    if ( ( Data is None ) and ( labels is None ) ):
        sys.exit( "[split__train_test.py] Data or labels == ??? [ERROR]" )
    
    # ------------------------------------------------- #
    # --- [2] split Data & labels, if exists        --- #
    # ------------------------------------------------- #
    if ( Data is not None ):
        Data_train , Data_test  = skms.train_test_split( Data  , train_size=train_size, \
                                                         shuffle=shuffle, \
                                                         random_state=random_state )
    if ( labels is not None ):
        label_train, label_test = skms.train_test_split( labels, train_size=train_size, \
                                                         shuffle=shuffle, \
                                                         random_state=random_state )
    # ------------------------------------------------- #
    # --- [3] reshape for images                    --- #
    # ------------------------------------------------- #
    if ( ( flatten ) and ( Data is not None ) ):
        Data_train = np.reshape( Data_train, (Data_train.shape[0],-1) )
        Data_test  = np.reshape( Data_test , (Data_test .shape[0],-1) )
        
    # ------------------------------------------------- #
    # --- [4] return                                --- #
    # ------------------------------------------------- #
    if   ( returnType.lower() in [ "list", "all", "train-test" ] ):
        return( [ Data_train, label_train, Data_test, label_test ] )
    elif ( returnType.lower() in [ "train" ] ):
        return( [ Data_train, label_train ] )
    elif ( returnType.lower() in [ "test" ]  ):
        return( [ Data_test, label_test ] )
    elif ( returnType.lower() in [ "dict" ]  ):
        return( { "Data_train" :Data_train , "Data_test" :Data_test , \
                  "label_train":label_train, "label_test":label_test } )
    else:
        print( "[train_test_split.py] returnType == {} ??? ".format( returnType ) )
        sys.exit()
        
    
# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #
if ( __name__=="__main__" ):

    x_, y_, z_  = 0, 1, 2
    
    import nkUtilities.equiSpaceGrid as esg
    x1MinMaxNum = [ 0.0, 1.0, 11 ]
    x2MinMaxNum = [ 0.0, 1.0, 11 ]
    x3MinMaxNum = [ 0.0, 0.0,  1 ]
    coord       = esg.equiSpaceGrid( x1MinMaxNum=x1MinMaxNum, x2MinMaxNum=x2MinMaxNum, \
                                     x3MinMaxNum=x3MinMaxNum, returnType = "point" )
    coord[:,z_] = np.sqrt( coord[:,x_]**2 + coord[:,y_]**2 )
    labels      = np.where( coord[:,z_] > 0.5, 1, 0 )

    dTr,lTr,dTe,lTe = split__train_test( Data=coord, labels=labels )
    print( dTr.shape, lTr.shape, dTe.shape, lTe.shape )
