import os
import torch
import numpy as np
import torch.utils.data as data
import random
from .build import DATASETS
import open3d as o3d
import open3d
from os import listdir
import logging
import copy


@DATASETS.register_module()
class crown(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')

        print(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()

        self.file_list = []
        for line in lines:
            line = line.strip()
            tax_id = line
            if 'Lower' in tax_id:
                taxonomy_id = '0'
            else:
                taxonomy_id = '1'
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': tax_id,
                'file_path': line
            })
        print(f'[DATASET] {len(self.file_list)} instances were loaded')
        
    def pc_norm(self, pc): 
        centroid = np.mean(pc, axis=0)
        std_pc = np.std(pc, axis=0)
        pc = (pc - centroid)/std_pc
        return pc,centroid,std_pc
      
    def normalize_points_mean_std(self,main,opposing,shell,marginline):
   
        new_context = copy.deepcopy(main)
        new_opposing = copy.deepcopy(opposing)
        new_crown = copy.deepcopy(shell)
        new_marginline = copy.deepcopy(marginline)

        context_mean, context_std = np.mean(np.concatenate((main.points, opposing.points), axis=0), axis=0), \
                                np.std(np.concatenate((main.points, opposing.points), axis=0), axis=0)
        # scale values
        new_context_points = (np.asarray(new_context.points) - context_mean) / context_std
        new_opposing_points = (np.asarray(opposing.points) - context_mean) / context_std
        new_crown_points = (np.asarray(shell.points) - context_mean) / context_std
        new_marginline_points = (np.asarray(marginline.points) - context_mean) / context_std
    
        return new_context_points, new_opposing_points, new_crown_points,new_marginline_points, context_mean, context_std    

    def __getitem__(self, idx):
    
        #read points
        sample = self.file_list[idx]
        
        for j in os.listdir(os.path.join(self.pc_path,sample['file_path'])):
            if 'Antagonist' in j:
               opposing = o3d.io.read_point_cloud(os.path.join(self.pc_path, sample['file_path'], j))
               #o3d.visualization.draw_geometries([opposing])
            if 'master' in j:
               main = o3d.io.read_point_cloud(os.path.join(self.pc_path, sample['file_path'], j))
               #o3d.visualization.draw_geometries([master])
            if 'shell' in j:
               shell= o3d.io.read_point_cloud(os.path.join(self.pc_path, sample['file_path'], j))
            if 'groundTruthMarginLine' in j:
               marginline= o3d.io.read_point_cloud(os.path.join(self.pc_path, sample['file_path'], j))   
               #o3d.visualization.draw_geometries([shell])
    
        
        #normalize
        main_only, opposing_only, shell, marginline_only = copy.deepcopy(main), copy.deepcopy(opposing), copy.deepcopy(shell),copy.deepcopy(marginline)
        
        main_only, opposing_only, shell ,marginline_only, centroid, std_pc = self.normalize_points_mean_std(main_only,opposing_only,shell,marginline_only)
        #sample from main
        patch_size_main=5120
        positive_main_idx = np.arange(len(main_only))
        try:
           positive_selected_main_idx = np.random.choice(positive_main_idx, size=patch_size_main, replace=False)
        except ValueError:
           positive_selected_main_idx = np.random.choice(positive_main_idx, size=patch_size_main, replace=True)   
        main_only_select = np.zeros([patch_size_main, main_only.shape[1]], dtype='float32')
        main_only_select[:] = main_only[positive_selected_main_idx, :]
        
        #sample from opposing
        patch_size_opposing=5120
        positive_opposing_idx = np.arange(len(opposing_only))
        try:
           positive_selected_opposing_idx = np.random.choice(positive_opposing_idx, size=patch_size_opposing, replace=False)
        except ValueError:
           positive_selected_opposing_idx = np.random.choice(positive_opposing_idx, size=patch_size_opposing, replace=True)   
        opposing_only_select = np.zeros([patch_size_opposing, opposing_only.shape[1]], dtype='float32')
        opposing_only_select[:] = opposing_only[positive_selected_opposing_idx, :]
        
        #sample from shell
        patch_size_shell=1568
        positive_shell_idx = np.arange(len(shell))
        try:
           positive_selected_shell_idx = np.random.choice(positive_shell_idx, size=patch_size_shell, replace=False)
        except ValueError:    
           positive_selected_shell_idx = np.random.choice(positive_shell_idx, size=patch_size_shell, replace=True)
        shell_select = np.zeros([patch_size_shell, shell.shape[1]], dtype='float32')
        shell_select[:] = shell[positive_selected_shell_idx, :]

        #sample from marginline
        patch_size_margin=1000
        positive_margin_idx = np.arange(len(marginline_only))
        try:
           positive_selected_margin_idx = np.random.choice(positive_margin_idx, size=patch_size_margin, replace=False)
        except ValueError:    
           positive_selected_margin_idx = np.random.choice(positive_margin_idx, size=patch_size_margin, replace=True)
        marginline_only_select = np.zeros([patch_size_margin, marginline_only.shape[1]], dtype='float32')
        marginline_only_select[:] = marginline_only[positive_selected_margin_idx, :]
        
        


        data_partial= torch.from_numpy(np.concatenate((main_only_select,marginline_only_select,opposing_only_select), axis=0)).float()
        data_gt=torch.from_numpy(np.concatenate((shell_select,marginline_only_select), axis=0)).float()

        value_centroid =torch.from_numpy(centroid).float()
        value_std_pc=torch.from_numpy(std_pc).float()


        return sample['taxonomy_id'], sample['model_id'], data_gt, data_partial,value_centroid,value_std_pc

    def __len__(self):
        return len(self.file_list)