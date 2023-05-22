import keras_tuner as kt
from ml_pipeline.train import build_hp_model
import time

tuner = kt.RandomSearch(
    build_hp_model,
    objective='val_categorical_accuracy',
    max_trials=10,
    overwrite=False
)

# get the best model
model = tuner.get_best_models()[0]

# save the model
model.save(f"models/model-{int(time.time())}.keras")
