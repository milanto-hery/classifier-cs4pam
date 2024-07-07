from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, Sequential
from keras.regularizers import l2


class CNNNetwork:
    
    def custom_cnn(self, INPUT_SHAPE):
        
        model_2d = Sequential()

        model_2d.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE))
        model_2d.add(MaxPooling2D(pool_size=(2, 2)))
        model_2d.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model_2d.add(MaxPooling2D(pool_size=(2, 2)))
        model_2d.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model_2d.add(MaxPooling2D(pool_size=(2, 2)))

        model_2d.add(Flatten())
        model_2d.add(Dense(256, activation='relu'))
        model_2d.add(Dropout(0.3))
        model_2d.add(Dense(units=2, activation='softmax'))

        model_2d.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                
        return model_2d
    
   
    def ResNet50(self, INPUT_SHAPE):
     # Weights are downloaded automatically when instantiating a model. ResNet 50 Top1 Accuracy is 0.749, Top 5 Accuracy is 0.921
        base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(INPUT_SHAPE)) 

        # Training Final Layer only
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation="relu", kernel_regularizer=l2(0.01))(x)
        predictions = Dense(2, activation="softmax")(x)

        # gets pretrained base_model input, predicts based on custom-dataset trained prediction
        model = Model(inputs=base_model.input, outputs=predictions)

        # don't touch pretrained base_model
        for layer in base_model.layers:
          layer.trainable = False

        # adamw optimizer is not available on keras, unfortunately :( 
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        return model

    def other_network(self):
        ''' Add other networks to this Class. '''
        
        None
