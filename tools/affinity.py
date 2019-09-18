import numpy as np
import os
import init_paths
import argparse
from models.sample_models import *
from task_viz import *
import scipy.io

parser = argparse.ArgumentParser()

parser.add_argument('--explain-result-root', dest='explain_result_root', type=str)
parser.set_defaults(explain_result_root='explain_result')

parser.add_argument('--dataset', dest='dataset', type=str)
parser.set_defaults(dataset='taskonomy')

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

attributions = {}
affinity = np.zeros((3, len(task_list), len(task_list)), float)

for method_key, explain_method in explain_methods.items():
    for target_task_i in range(len(task_list)):
        target_attribution = np.load(os.path.join(result_root, task_list[target_task_i], explain_method + '.npy'))
        target_attribution = np.abs(target_attribution)
        target_attribution = np.mean(target_attribution, axis=3)
        target_attribution = np.squeeze(target_attribution)
        f_norm = np.linalg.norm(target_attribution, axis=(1, 2))
        for index in range(len(target_attribution)):
            target_attribution[index] /= f_norm[index]
            target_attribution[index] = target_attribution[index] > np.mean(target_attribution[index])

        for source_task_i in range(len(task_list)):
            if source_task_i == target_task_i:
                affinity[method_index_mapping[method_key], target_task_i, source_task_i] = len(target_attribution)
            if source_task_i > target_task_i:
                continue

            source_attribution = np.load(
                os.path.join(result_root, task_list[source_task_i], explain_method + '.npy'))
            source_attribution = np.abs(source_attribution)
            source_attribution = np.mean(source_attribution, axis=3)
            source_attribution = np.squeeze(source_attribution)
            f_norm = np.linalg.norm(source_attribution, axis=(1, 2))
            for index in range(len(source_attribution)):
                source_attribution[index] /= f_norm[index]
                source_attribution[index] = source_attribution[index] > np.mean(source_attribution[index])
            source_target_op = np.logical_xor(source_attribution, target_attribution)
            source_target_op = np.logical_not(source_target_op)
            affinity[method_index_mapping[method_key], target_task_i, source_task_i] = np.sum(source_target_op)
            affinity[method_index_mapping[method_key], source_task_i, target_task_i] = np.sum(source_target_op)
        print('Target task {} done.'.format(task_list[target_task_i]))

np.save(os.path.join(prj_dir, args.explain_result_root, args.dataset, 'affinity.npy'), affinity / len(target_attribution))
scipy.io.savemat(os.path.join(prj_dir, args.explain_result_root, args.dataset, 'affinity.mat'), {'affinity': affinity / len(target_attribution)})

for task_i in range(len(task_list)):
    print('---------------- For task {} ----------------'.format(task_list[task_i]))
    for method, ind in method_index_mapping.items():
        print('-----> Method: {}'.format(method))
        affinity_ = affinity[ind][task_i]
        ind_sort = np.argsort(affinity_)
        print([task_list[t] for t in ind_sort])

