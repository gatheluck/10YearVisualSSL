"""Ordered, byte-bound dataset membership for completed-epoch continuation."""
from torch.utils.data import Subset

from downstream.revised_resume import hash_membership


def dataset_membership(dataset, task):
    """Hash actual reader inputs, never consume samples or persist private paths."""
    indices = list(range(len(dataset)))
    while isinstance(dataset, Subset):
        indices = [dataset.indices[i] for i in indices]
        dataset = dataset.dataset
    if task not in ('imagenet', 'ade20k', 'nyuv2', 'ssv2', 'coco'):
        raise ValueError('unknown revised dataset membership')

    def rows():
        for index in indices:
            if task == 'imagenet':
                path, label = dataset.samples[index]
                files, metadata = [path], [int(label), dataset.class_to_idx]
            elif task == 'ade20k':
                files, metadata = [dataset.images[index], dataset.masks[index]], None
            elif task == 'nyuv2':
                files = [dataset.mat_path, dataset.mat_path.parent / 'splits.mat']
                metadata = int(dataset.indices[index])
            elif task == 'ssv2':
                path, label = dataset.items[index]
                files = [path, *dataset.label_files]
                metadata = [int(label), dataset.class_to_idx]
            else:
                identifier = dataset.ids[index]
                path = dataset.img_dir / dataset.coco.loadImgs(identifier)[0]['file_name']
                files = [path, dataset.annotation_file]
                metadata = [int(identifier), dataset.category_to_label]
            yield {'files': files, 'metadata': metadata}
    return hash_membership(rows())
