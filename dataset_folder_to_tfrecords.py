import argparse
import os
from time import time

from PIL import Image, ImageOps
import numpy as np
import yaml
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

import util


def _get_image_filenames_and_labels(directory, map_label, train_test_ratio, random_seed):
    filenames = []
    labels = []

    for subdir, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.jpg'):
                subdir = subdir.replace('\\', '/')
                label = map_label[subdir.split('/')[-1]]

                filename = os.path.join(subdir, filename)
                filenames.append(filename)
                labels.append(label)
    
    rus = RandomUnderSampler(random_state=random_seed)
    X_res, y_res = rus.fit_resample(np.array(filenames).reshape(-1, 1), np.array(labels).reshape(-1, 1))
    filenames = X_res.flatten().tolist()
    labels = y_res.flatten().tolist()

    images = []
    for filename in filenames:
        img = np.array(ImageOps.grayscale(Image.open(filename)))
        img = img.reshape(img.shape[0], img.shape[1], 1)
        images.append(img)

    images = np.array(images)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=1.0-train_test_ratio, random_state=random_seed, stratify=labels, shuffle=True)

    return X_train, np.array(y_train), X_test, np.array(y_test)

def create_tfrecord(config_filename):
    config = yaml.safe_load(open(config_filename, 'r', encoding='utf-8'))
    info_dict = {'dataset': f'custom_dataset'}

    random_seed = config['random_seed']
    if random_seed is None:
        random_seed = int(time())

    np.random.seed(random_seed)  # Choose random seed
    info_dict['seed'] = random_seed


    train_imgs, train_labels, test_imgs, test_labels = _get_image_filenames_and_labels(config['dataset_input_path'], config['labels'], 
                                                                                            config['train_test_ratio'], random_seed)
    
    if config['limit_data']:
        size = config['limit_data']
    else:
        size = len(train_labels)

    train_imgs, train_labels, valid_imgs, valid_labels = util.split_dataset(
        images=train_imgs, labels=train_labels, num_classes=len(config['labels']),
        valid_ratio=config['valid_ratio'], limit=size)

    # Calculate mean of training dataset (does not include validation!)
    train_img_mean = util.calculate_mean(train_imgs)

    output_path = os.path.join(config['dataset_input_path'], config['dataset_output_path'])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        raise OSError('Directory already exists!')

    # Save it as a numpy array
    np.savez_compressed(os.path.join(output_path, f"{info_dict['dataset']}_train_mean"),
                        train_img_mean=train_img_mean)


    output_file = os.path.join(output_path, 'train_1.tfrecords')
    util.convert_to_tfrecords(train_imgs, train_labels, output_file)
    output_file = os.path.join(output_path, 'valid_1.tfrecords')
    util.convert_to_tfrecords(valid_imgs, valid_labels, output_file)
    output_file = os.path.join(output_path, 'test_1.tfrecords')
    util.convert_to_tfrecords(test_imgs, test_labels, output_file)

    info_dict['train_records'] = len(train_labels)
    info_dict['valid_records'] = len(valid_labels)
    info_dict['test_records'] = len(test_labels)
    info_dict['shape'] = list(train_imgs.shape[1:])

    util.create_info_file(out_path=output_path, info_dict=info_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config file')

    args = parser.parse_args()

    create_tfrecord(args.config)