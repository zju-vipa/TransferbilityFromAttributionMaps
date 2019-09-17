import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from scipy.special import comb


def baseline():
    all_task = 19
    rel = 5
    nonrel = all_task - rel

    precision = []
    recall = []

    for fetch in range(1, 20):
        sum = 0
        for i in range(1, 6):
            sum += comb(rel, i) * comb(nonrel, fetch - i) * (i / fetch)
        sum /= comb(all_task, fetch)
        recall_one = sum * fetch / 5.0
        precision.append(sum)
        recall.append(recall_one)

    return precision, recall


def preprocess(matrix):
    # delete 'Colorization' and 'Inpainting' (not target)
    mat = np.delete(matrix, (7, 19), axis=1)
    return mat


parser = argparse.ArgumentParser()

parser.add_argument('--explain-result-root', dest='explain_result_root', type=str)
parser.set_defaults(explain_result_root='explain_result')
args = parser.parse_args()

prj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
explain_methods = {'saliency': 'saliency', 'grad*input': 'gradXinput', 'elrp': 'elrp'}
method_index_mapping = {'saliency': 0, 'grad*input': 1, 'elrp': 2}

list_of_tasks = 'autoencoder curvature denoise edge2d edge3d \
keypoint2d keypoint3d colorization \
reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point \
segmentsemantic class_1000 class_places inpainting_whole'
task_list = list_of_tasks.split(' ')

explain_result = args.explain_result_root
affinity_taskonomy = np.load(os.path.join(prj_dir, explain_result, 'taskonomy', 'affinity.npy'))
affinity_coco = np.load(os.path.join(prj_dir, explain_result, 'coco', 'affinity.npy'))
affinity_indoor = np.load(os.path.join(prj_dir, explain_result, 'indoor', 'affinity.npy'))
affinity_gt = np.load(os.path.join(prj_dir, explain_result, 'sort_gt.npy'))

affinity_taskonomy = preprocess(affinity_taskonomy)
affinity_coco = preprocess(affinity_coco)
affinity_indoor = preprocess(affinity_indoor)

aff_dict = {'taskonomy': affinity_taskonomy, 'coco': affinity_coco, 'Indoor': affinity_indoor}
pr_dict = {}

affinity_gt_rel = affinity_gt[:, 1:6]
for dataset_k, aff_v in aff_dict.items():
    print('Which Dataset:{}'.format(dataset_k))
    for method, ind in method_index_mapping.items():
        print('Which Method: {}'.format(method))
        affinity_oneMethod = aff_v[ind]
        ind_sort = np.argsort(-affinity_oneMethod, axis=1)

        def pr(gt_matrix, test_matrix):
            k = test_matrix.shape[1]
            num_intersect = 0
            for i in range(test_matrix.shape[0]):
                array_gt = gt_matrix[i].squeeze()
                array_test = test_matrix[i].squeeze()
                num_intersect += len(np.intersect1d(array_gt, array_test))
            precision = num_intersect / k / 18
            recall = num_intersect / 5 / 18
            return precision, recall
        
        p_list = []
        r_list = []
        for k in range(1, 20):
            test_matrix = ind_sort[:, 1:k+1]
            precision_temp, recall_temp = pr(affinity_gt_rel, test_matrix)
            p_list.append(precision_temp)
            r_list.append(recall_temp)
        p = np.array(p_list).reshape(1, -1)
        r = np.array(r_list).reshape(1, -1)
        p_r = np.concatenate((p, r), axis=0)
        pr_dict['{}_{}'.format(dataset_k, method)] = p_r

precision_base, recall_base = baseline()
x_axis = "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19".split()

p_list, r_list = [], []
for k in range(1, 20):
    test_matrix_o = affinity_gt[:, 1:k+1]
    precision_oracle, recall_oracle = pr(affinity_gt_rel, test_matrix_o)
    p_list.append(precision_oracle)
    r_list.append(recall_oracle)
precision_oracle = np.array(p_list).reshape(1, -1)
recall_oracle = np.array(r_list).reshape(1, -1)
p_r = np.concatenate((precision_oracle, recall_oracle), axis=0)
pr_dict['oracle'] = p_r


plt.figure(figsize=(15, 13))
plt.tick_params(labelsize=25)
lines_p = plt.plot(x_axis, pr_dict['taskonomy_saliency'][0],
                   x_axis, pr_dict['taskonomy_grad*input'][0],
                   x_axis, pr_dict['taskonomy_elrp'][0],
                   x_axis, pr_dict['coco_saliency'][0],
                   x_axis, pr_dict['coco_grad*input'][0],
                   x_axis, pr_dict['coco_elrp'][0],
                   x_axis, pr_dict['Indoor_saliency'][0],
                   x_axis, pr_dict['Indoor_grad*input'][0],
                   x_axis, pr_dict['Indoor_elrp'][0],
                   x_axis, precision_base,
                   x_axis, pr_dict['oracle'][0])

plt.setp(lines_p[0], color='lightcoral', linewidth=2, linestyle='-', marker='^', markersize=12, mec='lightcoral')
plt.setp(lines_p[1], color='lawngreen', linewidth=2, linestyle='-', marker='o', markersize=12, mec='lawngreen')
plt.setp(lines_p[2], color='yellowgreen', linewidth=2, linestyle='-', marker='v', markersize=12, mec='yellowgreen')
plt.setp(lines_p[3], color='yellow', linewidth=2, linestyle='-', marker='>', markersize=12, mec='yellow')
plt.setp(lines_p[4], color='skyblue', linewidth=2, linestyle='-', marker='<', markersize=12, mec='skyblue')
plt.setp(lines_p[5], color='cyan', linewidth=2, linestyle='-', marker='*', markersize=12, mec='cyan')
plt.setp(lines_p[6], color='orange', linewidth=2, linestyle='-', marker='s', markersize=12, mec='orange')
plt.setp(lines_p[7], color='violet', linewidth=2, linestyle='-', marker='p', markersize=12, mec='violet')
plt.setp(lines_p[8], color='purple', linewidth=2, linestyle='-', marker='x', markersize=12, mec='purple')
plt.setp(lines_p[9], color='black', linewidth=2, linestyle='-', marker='D', markersize=12, mec='black')
plt.setp(lines_p[10], color='red', linewidth=2, linestyle='-', marker='H', markersize=12, mec='red')

plt.legend(('taskonomy_saliency',
            'taskonomy_grad*input',
            'taskonomy_elrp',
            'coco_saliency',
            'coco_grad*input',
            'coco_elrp',
            'indoor_saliency',
            'indoor_grad*input',
            'indoor_elrp',
            'random ranking',
            'oracle',), loc='best', prop={'size': 28})
plt.title('P@K Curve', {'size': 40})
plt.xlabel('K', {'size': 40})
plt.ylabel('Precision', {'size': 40})
plt.savefig('../fig/Precision-K-Curve.pdf', dpi=1200)


plt.figure(figsize=(15, 13))
plt.tick_params(labelsize=25)
lines_r = plt.plot(x_axis, pr_dict['taskonomy_saliency'][1],
                   x_axis, pr_dict['taskonomy_grad*input'][1],
                   x_axis, pr_dict['taskonomy_elrp'][1],
                   x_axis, pr_dict['coco_saliency'][1],
                   x_axis, pr_dict['coco_grad*input'][1],
                   x_axis, pr_dict['coco_elrp'][1],
                   x_axis, pr_dict['Indoor_saliency'][1],
                   x_axis, pr_dict['Indoor_grad*input'][1],
                   x_axis, pr_dict['Indoor_elrp'][1],
                   x_axis, precision_base,
                   x_axis, pr_dict['oracle'][1],)

plt.setp(lines_r[0], color='lightcoral', linewidth=2, linestyle='-', marker='^', markersize=12, mec='lightcoral')
plt.setp(lines_r[1], color='lawngreen', linewidth=2, linestyle='-', marker='o', markersize=12, mec='lawngreen')
plt.setp(lines_r[2], color='yellowgreen', linewidth=2, linestyle='-', marker='v', markersize=12, mec='yellowgreen')
plt.setp(lines_r[3], color='yellow', linewidth=2, linestyle='-', marker='>', markersize=12, mec='yellow')
plt.setp(lines_r[4], color='skyblue', linewidth=2, linestyle='-', marker='<', markersize=12, mec='skyblue')
plt.setp(lines_r[5], color='cyan', linewidth=2, linestyle='-', marker='*', markersize=12, mec='cyan')
plt.setp(lines_r[6], color='orange', linewidth=2, linestyle='-', marker='s', markersize=12, mec='orange')
plt.setp(lines_r[7], color='violet', linewidth=2, linestyle='-', marker='p', markersize=12, mec='violet')
plt.setp(lines_r[8], color='purple', linewidth=2, linestyle='-', marker='x', markersize=12, mec='purple')
plt.setp(lines_r[9], color='black', linewidth=2, linestyle='-', marker='D', markersize=12, mec='black')
plt.setp(lines_r[10], color='red', linewidth=2, linestyle='-', marker='H', markersize=12, mec='red')

plt.legend(('taskonomy_saliency',
            'taskonomy_grad*input',
            'taskonomy_elrp',
            'coco_saliency',
            'coco_grad*input',
            'coco_elrp',
            'indoor_saliency',
            'indoor_grad*input',
            'indoor_elrp',
            'random ranking',
            'oracle',), loc='best', prop={'size':28})
plt.title('R@K Curve', {'size': 40})
plt.xlabel('K', {'size': 40})
plt.ylabel('Recall', {'size': 40})
plt.savefig('../fig/Recall-K-Curve.pdf', dpi=1200)


'''
plt.figure(figsize=(15, 13))
plt.tick_params(labelsize=25)
lines_pr = plt.plot(pr_dict['taskonomy_saliency'][1], pr_dict['taskonomy_saliency'][0],
                    pr_dict['taskonomy_grad*input'][1], pr_dict['taskonomy_grad*input'][0],
                    pr_dict['taskonomy_elrp'][1], pr_dict['taskonomy_elrp'][0],
                    pr_dict['coco_saliency'][1], pr_dict['coco_saliency'][0],
                    pr_dict['coco_grad*input'][1], pr_dict['coco_grad*input'][0],
                    pr_dict['coco_elrp'][1], pr_dict['coco_elrp'][0],
                    pr_dict['Indoor_saliency'][1], pr_dict['Indoor_saliency'][0],
                    pr_dict['Indoor_grad*input'][1], pr_dict['Indoor_grad*input'][0],
                    pr_dict['Indoor_elrp'][1], pr_dict['Indoor_elrp'][0],
                    recall_base, precision_base,
                    pr_dict['oracle'][1], pr_dict['oracle'][0])

plt.setp(lines_pr[0], color='lightcoral', linewidth=2, linestyle='-', marker='^', markersize=12, mec='lightcoral')
plt.setp(lines_pr[1], color='lawngreen', linewidth=2, linestyle='-', marker='o', markersize=12, mec='lawngreen')
plt.setp(lines_pr[2], color='yellowgreen', linewidth=2, linestyle='-', marker='v', markersize=12, mec='yellowgreen')
plt.setp(lines_pr[3], color='yellow', linewidth=2, linestyle='-', marker='>', markersize=12, mec='yellow')
plt.setp(lines_pr[4], color='skyblue', linewidth=2, linestyle='-', marker='<', markersize=12, mec='skyblue')
plt.setp(lines_pr[5], color='cyan', linewidth=2, linestyle='-', marker='*', markersize=12, mec='cyan')
plt.setp(lines_pr[6], color='orange', linewidth=2, linestyle='-', marker='s', markersize=12, mec='orange')
plt.setp(lines_pr[7], color='violet', linewidth=2, linestyle='-', marker='p', markersize=12, mec='violet')
plt.setp(lines_pr[8], color='purple', linewidth=2, linestyle='-', marker='x', markersize=12, mec='purple')
plt.setp(lines_pr[9], color='black', linewidth=2, linestyle='-', marker='D', markersize=12, mec='black')
plt.setp(lines_pr[10], color='red', linewidth=2, linestyle='-', marker='H', markersize=12, mec='red')

plt.legend(('taskonomy_saliency',
            'taskonomy_grad*input',
            'taskonomy_elrp',
            'coco_saliency',
            'coco_grad*input',
            'coco_elrp',
            'indoor_saliency',
            'indoor_grad*input',
            'indoor_elrp',
            'baseline',
            'oracle',), loc='best', prop={'size': 25})

plt.title('PR Curve', {'size': 35})
plt.xlabel('Recall', {'size': 35})
plt.ylabel('Precision', {'size': 35})
plt.savefig('../fig/Precision-Recall-Curve.pdf', dpi=1200)
'''
