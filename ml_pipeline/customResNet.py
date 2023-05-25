from tensorflow import keras
from keras import layers
from keras_tuner import RandomSearch

class CustomResNet:
    # target resnet-22 depth

    def __init__(self, inputs, num_classes, hp_dict=None):
        self.inputs = inputs
        self.num_classes = num_classes
        self.hp_dict = hp_dict

    def identity_block(self, X, f, filters: tuple, stage, block):
        """
        Implementation of the identity block as defined in Figure 3
        
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        
        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """
        
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # filter/unit sizes
        F1, F2, F3 = filters

        X_shortcut = X

        # Save the input value. You'll need this later to add back to the main path. 
        X_shortcut = X
        
        # First component of main path
        X = layers.Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a')(X)
        X = layers.BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
        X = layers.Activation('relu')(X)

        # Second component of main path (≈3 lines)
        X = layers.Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b')(X)
        X = layers.BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
        X = layers.Activation('relu')(X)

        # Third component of main path (≈2 lines)
        X = layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c')(X)
        X = layers.BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = layers.Add()([X, X_shortcut])
        X = layers.Activation('relu')(X)

        return X

    def convolutional_block(self, X, f, filters, stage, block, s = 2):
        """
        Implementation of the convolutional block as defined in Figure 4
        
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        s -- Integer, specifying the stride to be used
        
        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """
        
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        # Retrieve Filters
        F1, F2, F3 = filters
        
        # Save the input value
        X_shortcut = X


        ##### MAIN PATH #####
        # First component of main path 
        X = layers.Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a')(X)
        X = layers.BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
        X = layers.Activation('relu')(X)

        # Second component of main path (≈3 lines)
        X = layers.Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b')(X)
        X = layers.BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
        X = layers.Activation('relu')(X)


        # Third component of main path (≈2 lines)
        X = layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c')(X)
        X = layers.BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)


        ##### SHORTCUT PATH #### (≈2 lines)
        X_shortcut = layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1')(X_shortcut)
        X_shortcut = layers.BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = layers.Add()([X, X_shortcut])
        X = layers.Activation('relu')(X)
        
        return X
    

    def build_model(self, hp=None):
        # normalise the image data
        scaler = layers.Rescaling(1./255)

        # scaling layer
        chain = scaler(self.inputs)

        # Zero-Padding
        chain = layers.ZeroPadding2D((3, 3))(chain)

        # convolutional layers
        chain = layers.Conv2D(
            32,
            6,
            name='conv1'
        )(chain)
        chain = layers.BatchNormalization(axis=3, name='bn_conv1')(chain)
        chain = layers.Activation('relu')(chain)
        chain = layers.MaxPooling2D()(chain)

        # Stage 2
        f = 10 if self.hp_dict == None \
            else hp.Int(
                "kernals1", 
                min_value=self.hp_dict["conv"]["kernal_min"], 
                max_value=self.hp_dict["conv"]["kernal_max"], 
                step=self.hp_dict["conv"]["kernal_step"]
            )
        filters = (
            64 if self.hp_dict == None 
            else hp.Int(
                "units1", 
                min_value=self.hp_dict["conv"]["unit_min"], 
                max_value=self.hp_dict["conv"]["unit_max"], 
                step=self.hp_dict["conv"]["unit_step"]
            ),
            64 if self.hp_dict == None 
            else hp.Int(
                "units2", 
                min_value=self.hp_dict["conv"]["unit_min"], 
                max_value=self.hp_dict["conv"]["unit_max"], 
                step=self.hp_dict["conv"]["unit_step"]
            ),
            256 if self.hp_dict == None  
            else hp.Int(
                "units3", 
                min_value=self.hp_dict["conv"]["unit_min"], 
                max_value=self.hp_dict["conv"]["unit_max"], 
                step=self.hp_dict["conv"]["unit_step"]
            )
        )
            
        
        chain = self.convolutional_block(chain, f=f, filters=filters, stage=2, block='a', s=1)
        chain = self.identity_block(chain, f, filters, stage=2, block='b')
        chain = self.identity_block(chain, f, filters, stage=2, block='c')

        # Stage 3 (≈3 lines)
        f1, f2, f3 = filters
        f1 = f1*2
        f2 = f2*2
        f3 = f3*2
        filters = (f1, f2, f3)

        chain = self.convolutional_block(chain, f = f*2, filters = filters, stage = 3, block='a', s = 2)
        chain = self.identity_block(chain, f*2, filters, stage=3, block='b')
        chain = self.identity_block(chain, f*2, filters, stage=3, block='c')

        # average pooling layer
        chain = layers.GlobalAveragePooling2D(name="avg_pool")(chain)

        # apply a dense classifier
        num_classes = self.num_classes
        chain = layers.Dense(
            num_classes**2 if self.hp_dict == None
            else num_classes ** hp.Int(
                "dense",
                min_value=self.hp_dict["dense"]["min_power"],
                max_value=self.hp_dict["dense"]["max_power"],
                step=self.hp_dict["dense"]["step"]
            ), 
            activation="relu"
        )(chain)

        # apply a final dense classifier
        outputs = layers.Dense(num_classes, activation="softmax")(chain)

        # return the outputs
        return outputs