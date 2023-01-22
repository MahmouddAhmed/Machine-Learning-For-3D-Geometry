from pathlib import Path
import json

import numpy as np
import torch


class ShapeNet(torch.utils.data.Dataset):
    num_classes = 8
    dataset_sdf_path = Path(
        "exercise_3/data/shapenet_dim32_sdf")  # path to voxel data
    # path to voxel data
    dataset_df_path = Path("exercise_3/data/shapenet_dim32_df")
    # mapping for ShapeNet ids -> names
    class_name_mapping = json.loads(
        Path("exercise_3/data/shape_info.json").read_text())
    classes = sorted(class_name_mapping.keys())

    def __init__(self, split):
        super().__init__()
        assert split in ['train', 'val', 'overfit']
        self.truncation_distance = 3

        # keep track of shapes based on split
        self.items = Path(
            f"exercise_3/data/splits/shapenet/{split}.txt").read_text().splitlines()

    def __getitem__(self, index):
        sdf_id, df_id = self.items[index].split(' ')

        input_sdf = ShapeNet.get_shape_sdf(sdf_id)
        target_df = ShapeNet.get_shape_df(df_id)

        # TODO Apply truncation to sdf and df
        input_sdf = input_sdf.clip(
            self.truncation_distance*-1, self.truncation_distance)
        target_df = target_df.clip(
            self.truncation_distance*-1, self.truncation_distance)
        # TODO Stack (distances, sdf sign) for the input sdf
        input_sdf = np.repeat(input_sdf[np.newaxis, :, :, :, ], 2, axis=0)
        input_sdf[0] = np.abs(input_sdf[0])
        input_sdf[1] = np.sign(input_sdf[1])

        # TODO Log-scale target df
        target_df = np.log(target_df+1)

        return {
            'name': f'{sdf_id}-{df_id}',
            'input_sdf': input_sdf,
            'target_df': target_df
        }

    def __len__(self):
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        # TODO add code to move batch to device
        batch['input_sdf'] = batch['input_sdf'].to(device)
        batch['target_df'] = batch['target_df'].to(device)

    @staticmethod
    def get_shape_sdf(shapenet_id):
        sdf = None
        # TODO implement sdf data loading
        file_path = str(ShapeNet.dataset_sdf_path)+'/'+shapenet_id+'.sdf'
        dimX, dimY, dimZ = np.fromfile(file_path, dtype=np.uint64, count=3)
        data = np.fromfile(file_path, dtype=np.float32,
                           count=dimX*dimY*dimZ, offset=24)
        sdf = data.reshape((dimX, dimY, dimZ))
        return sdf

    @staticmethod
    def get_shape_df(shapenet_id):
        df = None
        # TODO implement df data loading
        file_path = str(ShapeNet.dataset_df_path)+'/'+shapenet_id+'.df'
        dimX, dimY, dimZ = np.fromfile(file_path, dtype=np.uint64, count=3)
        data = np.fromfile(file_path, dtype=np.float32,
                           count=dimX*dimY*dimZ, offset=24)
        df = data.reshape((dimX, dimY, dimZ))
        return df
