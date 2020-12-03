from argparse import ArgumentParser
import sys
import os

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-model', action='store', dest='model_name', type=str)
    parser.add_argument('-dataset', action='store', dest='dataset_location', type=str)
    parser.add_argument('-save', action='store', dest='save_location', type=str)
    parser.add_argument('-mode', action='store', dest='mode', type=str)
    parser.add_argument('-weights', action='store', dest='weights_location', type=str)
    flags = parser.parse_args()

    os.environ['NASA_DATASET'] = flags.dataset_location
    os.environ['WEIGHTS_SAVE_LOCATION'] = flags.save_location

    if flags.model_name == 'hierarchial':
        if flags.mode == 'test':
            os.system('python src/hierarchial/test.py %s' % flags.weights_location)
        else:
            os.system('python src/hierarchial/train.py')
    elif flags.model_name == 'fcvae':
        raise NotImplementedError

