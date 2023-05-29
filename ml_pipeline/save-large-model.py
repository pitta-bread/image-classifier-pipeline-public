import keras_tuner as kt
from ml_pipeline.train import build_hp_model
import time, datetime

tuner = kt.RandomSearch(
    build_hp_model,
    objective='val_categorical_accuracy',
    max_trials=10,
    overwrite=False,
    project_name='customResNet-2-20230526'
)

destination = "models"

print(tuner.get_best_hyperparameters())

# get the best model
model = tuner.get_best_models()[0]

# save the model
# serialize model to JSON
model_json = model.to_json()
with open(f"{destination}/8-weights-resnet-model-{int(time.time())}.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights(f"{destination}/8-weights-resnet-model-{int(time.time())}.h5")
print("Saved model to disk")
