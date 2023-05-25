from tensorflow import keras
from keras import layers
from keras_tuner import RandomSearch

class CustomResNet:

    def __init__(self, inputs, num_classes, hp_dict=None):
        self.inputs = inputs
        self.num_classes = num_classes
        self.hp_dict = hp_dict

    def identity_block(self, X, f, filters, stage, block):
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


    

    def build_model(self, hp=None):
        # normalise the image data
        scaler = layers.Rescaling(1./255)

        # scaling layer
        chain = scaler(self.inputs)

        # convolutional layers
        chain = layers.Conv2D(
            32 if self.hp_dict == None 
            else hp.Int(
                "units1", 
                min_value=self.hp_dict["conv"]["unit_min"], 
                max_value=self.hp_dict["conv"]["unit_max"], 
                step=self.hp_dict["conv"]["unit_step"]
            ),
            10 if self.hp_dict == None 
            else hp.Int(
                "kernals1", 
                min_value=self.hp_dict["conv"]["kernal_min"], 
                max_value=self.hp_dict["conv"]["kernal_max"], 
                step=self.hp_dict["conv"]["kernal_step"]
            ),
            activation="relu"
        )(chain)
        chain = layers.MaxPooling2D()(chain)

        chain = layers.Conv2D(
            32 if self.hp_dict == None 
            else hp.Int(
                "units2", 
                min_value=self.hp_dict["conv"]["unit_min"], 
                max_value=self.hp_dict["conv"]["unit_max"], 
                step=self.hp_dict["conv"]["unit_step"]
            ),
            10 if self.hp_dict == None 
            else hp.Int(
                "kernals2", 
                min_value=self.hp_dict["conv"]["kernal_min"], 
                max_value=self.hp_dict["conv"]["kernal_max"], 
                step=self.hp_dict["conv"]["kernal_step"]
            ),
            activation="relu"
        )(chain)
        chain = layers.MaxPooling2D()(chain)

        chain = layers.Conv2D(
            32 if self.hp_dict == None  
            else hp.Int(
                "units3", 
                min_value=self.hp_dict["conv"]["unit_min"], 
                max_value=self.hp_dict["conv"]["unit_max"], 
                step=self.hp_dict["conv"]["unit_step"]
            ),
            10 if self.hp_dict == None 
            else hp.Int(
                "kernals3", 
                min_value=self.hp_dict["conv"]["kernal_min"], 
                max_value=self.hp_dict["conv"]["kernal_max"], 
                step=self.hp_dict["conv"]["kernal_step"]
            ),
            activation="relu"
        )(chain)
        chain = layers.MaxPooling2D()(chain)

        # apply global average pooling to get a 1D feature vector
        chain = layers.GlobalAveragePooling2D()(chain)

        # dropout layer
        chain = layers.Dropout(
            0.5 if self.hp_dict == None
            else hp.Float(
                "dropout",
                min_value=self.hp_dict["dropout"]["min"],
                max_value=self.hp_dict["dropout"]["max"],
                step=self.hp_dict["dropout"]["step"]
            )
        )(chain)

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