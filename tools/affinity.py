# -*- coding: utf-8 -*-
import numpy as np
import os
import init_paths
import argparse
import time
import scipy.io

parser = argparse.ArgumentParser()

parser.add_argument('--explain-result-root', dest='explain_result_root', type=str)
parser.set_defaults(explain_result_root='explain_result')

parser.add_argument('--dataset', dest='dataset', type=str)
parser.set_defaults(dataset='taskonomy')

parser.add_argument('--imlist-size', dest='imlist_size', type=int)
parser.set_defaults(imlist_size=1000)

args = parser.parse_args()

prj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
result_root = os.path.join(prj_dir, args.explain_result_root, args.dataset)
explain_methods = {'saliency': 'saliency', 'grad*input': 'gradXinput', 'elrp': 'elrp'}
method_index_mapping = {'saliency': 0, 'grad*input': 1, 'elrp': 2}

list_of_tasks = 'autoencoder curvature denoise edge2d edge3d \
keypoint2d keypoint3d colorization \
reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point \
segmentsemantic class_1000 class_places inpainting_whole'
task_list = list_of_tasks.split(' ')

affinity = np.zeros((3, len(task_list), len(task_list)), float)
attribution_all = np.zeros((3, len(task_list), args.imlist_size, 256*256))

for method_key, explain_method in explain_methods.items():
    for task_i in range(len(task_list)):
        attribution = np.load(os.path.join(result_root, task_list[task_i], explain_method + '.npy'))
        attribution = np.mean(attribution, axis=3)
        attribution = attribution.reshape(attribution.shape[0], -1)
        attribution_all[method_index_mapping[method_key], task_i] = attribution

for method_key, explain_method in explain_methods.items():
    for target_task_i in range(len(task_list)):
        for source_task_i in range(len(task_list)):
            if source_task_i == target_task_i:
                affinity[method_index_mapping[method_key], target_task_i, source_task_i] = args.imlist_size
                continue
            if source_task_i > target_task_i:
                continue

            target_attribution = attribution_all[method_index_mapping[method_key], target_task_i]
            source_attribution = attribution_all[method_index_mapping[method_key], source_task_i]

            affinity_sum = 0
            for ind in range(target_attribution.shape[0]):
                affinity_sum += np.inner(target_attribution[ind], source_attribution[ind]) / \
                                (np.linalg.norm(target_attribution[ind])*np.linalg.norm(source_attribution[ind]))

            affinity[method_index_mapping[method_key], target_task_i, source_task_i] = affinity_sum
            affinity[method_index_mapping[method_key], source_task_i, target_task_i] = affinity_sum

        print('Target task {} done.'.format(task_list[target_task_i]))

np.save(os.path.join(prj_dir, args.explain_result_root, args.dataset, 'affinity.npy'), affinity / args.imlist_size)
scipy.io.savemat(os.path.join(prj_dir, args.explain_result_root, args.dataset, 'affinity.mat'),
                  {'affinity': affinity / args.imlist_size})

for task_i in range(len(task_list)):
    print('---------------- For task {} ----------------'.format(task_list[task_i]))
    for method, ind in method_index_mapping.items():
        print('-----> Method: {}'.format(method))
        affinity_ = affinity[ind][task_i]
        ind_sort = np.argsort(affinity_)
        print([task_list[t] for t in ind_sort])

