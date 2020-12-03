# nasa-anomaly

## Install

```sh
pip install -r requirements.txt
```

## Train models

```sh
python main.py -model <model_name> -dataset <path_to_dataset> -save <path_to_save_weights_and_losses>
```

An example might be,

```sh
python main.py -model hierarchial -dataset drive/MyDrive/DASHlink_multiclass_all_ML.npz -save .
```

## Infer models

```sh
python main.py -model hierarchial -mode test -dataset <path_to_dataset> -weights <path_to_the_weights_location>
```

## Artififacts

The folder named `/artifacts` contains all the plots, losses, and weights generated by the model in the poster.