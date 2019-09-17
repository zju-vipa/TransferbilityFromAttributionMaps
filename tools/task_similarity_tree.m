clear,clc

load('../explain_result/taskonomy/affinity.mat');
affinity_saliency = affinity(1);
affinity_gradxinput = affinity(2);
affinity_elrp = affinity(3);

task_list = {'Autoencoder', 'Curvature', 'Denoise', 'Edge 2D', 'Edge 3D', ...
'Keypoint 2D','Keypoint 3D', ...
'Reshade' ,'Rgb2depth' ,'Rgb2mist','Rgb2sfnorm', ...
'Room Layout', 'Segment 25D', 'Segment 2D', 'Vanishing Point', ...
'Segment Semantic' ,'Class 1000' ,'Class Places'};

plot_dendrogram(affinity_saliency, task_list);
plot_dendrogram(affinity_gradxinput, task_list);
plot_dendrogram(affinity_elrp, task_list);
