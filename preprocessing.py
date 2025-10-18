import os
import numpy as np
from branched_resnet_v2 import CustomImageDataset, dataset_load

def import_data(directory, save_path=None, save=False):
    """
    Import data from a directory of .npz files.
    """
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.npz'):
            file_path = os.path.join(directory, filename)
            loaded_data = np.load(file_path)
            data.append(loaded_data)

    # concatenate the data from all files
    all_data = {}
    for key in data[0].keys():
        all_data[key] = np.concatenate([d[key] for d in data], axis=0)

    # check the shape of the concatenated data
    for key, value in all_data.items():
        print(f"{key}: {value.shape}")  

    if save:
        if save_path is None:
            save_path = f'datasets/{directory}_concatenated_data.npz'
        np.savez(save_path, **all_data)
        print(f"Data saved to {save_path}")

    return all_data

def normalize_image(image, mean=0.5, std=0.5):
    """
    Normalize an image tensor to have a mean and standard deviation.
    """
    return (image - mean) / std

def normalize_images(images, mean=0.5, std=0.5):
    """
    Normalize a list of images.
    """
    return [normalize_image(image, mean, std) for image in images]

def shuffle_data(data, seed=42):

    np.random.seed(seed)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    return data[indices]

def combine_npzs(data_dir):
    combined_data = {}
    order = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'last']
    # Process files in the order specified by the 'order' list
    for pos in order:
        for filename in os.listdir(data_dir):
            file_split = filename.split('_')
            if pos in file_split and filename.endswith('.npz'):
                print(f"Processing file: {filename}")
                file_path = os.path.join(data_dir, filename)
                data = np.load(file_path)
                for key in data.files:
                    if key in combined_data:
                        combined_data[key] = np.concatenate((combined_data[key], data[key]), axis=0)
                    else:
                        combined_data[key] = data[key]
    return combined_data

def combine_data(orig_data, new_data):
    combined_data = dict(orig_data)

    combined_data['Ring_Artifact_v1'] = new_data['Ring_Artifact_v1']
    combined_data['ring_labels'] = new_data['label']

    return combined_data

def preprocess_data(data, distortions, include_original=True, save_data = False, save_path=None):

    keys = list(data.keys())

    if 'Ring_Artifact_v1' in distortions:
        ring_flag = True
        distortions.remove('Ring_Artifact_v1')
    else:
        ring_flag = False

    if include_original:
        images = [data[keys[0]]]
    else:
        images = []

    for distortion in distortions:
        images.append(data[distortion])

    labels = data[keys[1]]
    ring_labels = data[keys[-1]]

    normalized_images = []
    for image in images:
        normalized_images.append(normalize_images(image))

    zero_labels = np.zeros_like(labels)
    one_labels = np.ones_like(labels)

    if include_original:
        domain_label_list = [zero_labels]
        expanded_label_list = [labels]
    else:
        domain_label_list = []
        expanded_label_list = []

    for _ in distortions:
        domain_label_list.append(one_labels)
        expanded_label_list.append(labels)

    if domain_label_list != []:
        domain_labels = np.concatenate(domain_label_list, axis=0)
        
    if expanded_label_list != []:
        expanded_labels = np.concatenate(expanded_label_list, axis=0)

    concatenated_images = np.concatenate(normalized_images, axis=0)

    if ring_flag:
        ring_images = data['Ring_Artifact_v1']
        ring_labels = data['ring_labels']
        ring_images = normalize_images(ring_images)
        concatenated_images = np.concatenate((concatenated_images, ring_images), axis=0)
        domain_labels = np.concatenate((domain_labels, one_labels), axis=0)
        expanded_labels = np.concatenate((expanded_labels, ring_labels), axis=0)


    print(len(concatenated_images), len(expanded_labels), len(domain_labels))
    assert len(concatenated_images) == len(expanded_labels) == len(domain_labels), "Dataset length mismatch!"

    # Shuffle the concatenated images and labels
    seed = 42
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(concatenated_images))
    concatenated_images = concatenated_images[shuffled_indices]
    expanded_labels = expanded_labels[shuffled_indices]
    domain_labels = domain_labels[shuffled_indices]

    if save_data:
        if save_path is None:
            raise ValueError("save_path must be specified if save_data is True")
        np.savez_compressed(save_path, images=concatenated_images, labels1=expanded_labels, labels2=domain_labels)

    dataset = CustomImageDataset(images=concatenated_images, labels1=expanded_labels, labels2=domain_labels)

    return dataset
