from __future__ import absolute_import, division, print_function

import argparse
import importlib
import numpy as np
import os
import tensorflow as tf
import init_paths
from models.sample_models import *
import skimage
import skimage.io
from task_viz import *
import random
import utils
import lib.data.load_ops as load_ops
from lib.deepexplain.tensorflow import DeepExplain
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
cur_dir = os.path.dirname(__file__)
prj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


parser = argparse.ArgumentParser()

parser.add_argument('--explain-result-root', dest='explain_result_root', type=str)
parser.set_defaults(explain_result_root='explain_result')

parser.add_argument('--dataset', dest='dataset', type=str)
parser.set_defaults(dataset='taskonomy')

parser.add_argument('--imlist-size', dest='imlist_size', type=int)
parser.set_defaults(imlist_size=1000)

tf.logging.set_verbosity(tf.logging.ERROR)

list_of_tasks = 'autoencoder curvature denoise edge2d edge3d \
keypoint2d keypoint3d colorization \
reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point \
segmentsemantic class_1000 class_places inpainting_whole'
list_of_tasks = list_of_tasks.split()


def generate_cfg(task):
    repo_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    CONFIG_DIR = os.path.join(repo_dir, 'experiments/final', task)
    ############## Load Configs ##############
    import utils
    from general_utils import RuntimeDeterminedEnviromentVars
    cfg = utils.load_config(CONFIG_DIR, nopause=True)
    RuntimeDeterminedEnviromentVars.register_dict(cfg)
    cfg['batch_size'] = 1
    if 'batch_size' in cfg['encoder_kwargs']:
        cfg['encoder_kwargs']['batch_size'] = 1
    cfg['model_path'] = os.path.join(repo_dir, 'temp', task, 'model.permanent-ckpt')
    cfg['root_dir'] = repo_dir
    return cfg


def deep_attribution():
    import general_utils
    from general_utils import RuntimeDeterminedEnviromentVars

    tf.logging.set_verbosity(tf.logging.ERROR)

    args = parser.parse_args()
    imlist_file_path = os.path.join(prj_dir, args.explain_result_root, args.dataset, 'imlist.txt')
    explain_result_root = os.path.join(prj_dir, args.explain_result_root, args.dataset)
    with open(imlist_file_path) as f:
        lines = []
        line = f.readline().strip()
        while len(line) != 0:
            lines += [os.path.join(prj_dir, 'dataset', args.dataset, line)]
            line = f.readline().strip()
    explain_methods = ['saliency', 'grad*input', 'elrp']
    for task in list_of_tasks:
        if not os.path.exists(os.path.join(explain_result_root, task)):
            os.mkdir(os.path.join(explain_result_root, task))
        cfg = generate_cfg(task)
        print("Doing {task}".format(task=task))
        general_utils = importlib.reload(general_utils)

        tf.reset_default_graph()
        sess = tf.Session()
        training_runners = {'sess': sess, 'coord': tf.train.Coordinator()}
        with DeepExplain(session=sess, graph=sess.graph) as de:
            ############## Set Up Inputs ##############
            setup_input_fn = utils.setup_input
            inputs = setup_input_fn(cfg, is_training=False, use_filename_queue=False)
            RuntimeDeterminedEnviromentVars.load_dynamic_variables(inputs, cfg)
            RuntimeDeterminedEnviromentVars.populate_registered_variables()
            ############## Set Up Model ##############
            model = utils.setup_model(inputs, cfg, is_training=False)
            m = model['model']
            model['saver_op'].restore(training_runners['sess'], cfg['model_path'])
            encoder_endpoints = model['model'].encoder_endpoints
            endpoints = encoder_endpoints

        print('There are {} images in {}'.format(len(lines), imlist_file_path))
        img = load_raw_image_center_crop(lines[0])
        img = skimage.img_as_float(img)
        low_sat_tasks = 'autoencoder curvature denoise edge2d edge3d \
                    keypoint2d keypoint3d \
                    reshade rgb2depth rgb2mist rgb2sfnorm \
                    segment25d segment2d room_layout'.split()
        if task in low_sat_tasks:
            cfg['input_preprocessing_fn'] = load_ops.resize_rescale_image_low_sat
        img = cfg['input_preprocessing_fn'](img, **cfg['input_preprocessing_fn_kwargs'])
        imgs = np.zeros([args.imlist_size, 1]+list(img.shape), float)
        for line_i in range(args.imlist_size):
            img = load_raw_image_center_crop(lines[line_i])
            img = skimage.img_as_float(img)
            if task not in list_of_tasks:
                raise ValueError('Task not supported')
            if task in low_sat_tasks:
                cfg['input_preprocessing_fn'] = load_ops.resize_rescale_image_low_sat

            if task == 'jigsaw':
                img = cfg['input_preprocessing_fn'](img, target=cfg['target_dict'][random.randint(0, 99)],
                                                    **cfg['input_preprocessing_fn_kwargs'])
            else:
                img = cfg['input_preprocessing_fn'](img, **cfg['input_preprocessing_fn_kwargs'])

            imgs[line_i, :] = img[np.newaxis, :]

        elrp = np.zeros([args.imlist_size] + list(img.shape), float)
        saliency = np.zeros([args.imlist_size] + list(img.shape), float)
        gradXinput = np.zeros([args.imlist_size] + list(img.shape), float)
        for im_i in range(args.imlist_size):
            with DeepExplain(session=sess) as de:
                representation = training_runners['sess'].run(
                    [endpoints['encoder_output']], feed_dict={m.input_images: imgs[im_i]})
                attributions = {
                explain_method: de.explain(explain_method, endpoints['encoder_output'], m.input_images, imgs[im_i]) for
                explain_method in explain_methods}
            elrp[im_i] = attributions['elrp']
            saliency[im_i] = attributions['saliency']
            gradXinput[im_i] = attributions['grad*input']
            print('{} images done.'.format(im_i))
            if ((im_i+1) % 5) == 0:
                ############## Clean Up ##############
                training_runners['coord'].request_stop()
                training_runners['coord'].join()
                ############## Reset graph and paths ##############
                tf.reset_default_graph()
                training_runners['sess'].close()
                sess = tf.Session()
                training_runners = {'sess': sess, 'coord': tf.train.Coordinator()}
                with DeepExplain(session=sess, graph=sess.graph) as de:
                    ############## Set Up Inputs ##############
                    setup_input_fn = utils.setup_input
                    inputs = setup_input_fn(cfg, is_training=False, use_filename_queue=False)
                    RuntimeDeterminedEnviromentVars.load_dynamic_variables(inputs, cfg)
                    RuntimeDeterminedEnviromentVars.populate_registered_variables()
                    ############## Set Up Model ##############
                    model = utils.setup_model(inputs, cfg, is_training=False)
                    m = model['model']
                    model['saver_op'].restore(training_runners['sess'], cfg['model_path'])
                    encoder_endpoints = model['model'].encoder_endpoints
                    endpoints = encoder_endpoints

        np.save(os.path.join(explain_result_root, task, 'elrp.npy'), elrp)
        np.save(os.path.join(explain_result_root, task, 'saliency.npy'), saliency)
        np.save(os.path.join(explain_result_root, task, 'gradXinput.npy'), gradXinput)
        ############## Clean Up ##############
        training_runners['coord'].request_stop()
        training_runners['coord'].join()
        ############## Reset graph and paths ##############
        tf.reset_default_graph()
        training_runners['sess'].close()
        print('Task {} Done!'.format(task))
    print('All Done.')
    return


if __name__ == '__main__':
    deep_attribution()

