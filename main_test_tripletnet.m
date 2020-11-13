clc;
clear;
close all;

magin = [0.1, 0.5, 1, 5];

for m = 1 : length(magin)
    if m ~= 1
        clear trainedNet
    end
    
    %% update mode
    update_mode = 'VGG_fix'; % 'VGG_fix' or 'VGG_update'
    
    toolbox_path = '..\..\..\toolbox\matconvnet\';
    run([toolbox_path 'matlab\vl_setupnn.m']);
    
    net_path = ['.\results\Tripletnet_', update_mode, '_magin_', num2str(magin(m))];
    trainedNet = load(fullfile(net_path, 'network_5.mat'));
    trainedNet = dagnn.DagNN.loadobj(trainedNet);
    layers = trainedNet.layers;
    for l = 1 : length(layers)
        layer_name = layers(l).name;
        if ~isempty(strfind(layer_name, 'same')) || ~isempty(strfind(layer_name, 'different'))% || isempty(strfind(layer_name, 'gen'))
            trainedNet.removeLayer(layer_name);
        end
    end
    trainedNet.removeLayer('triplet_loss');
    trainedNet.conserveMemory = 0;
    out_middle = 'x40';
    gpu_mode = 1;
    save_path = ['.\results\Tripletnet_', update_mode, '_magin_', num2str(magin(m))];
    % load triplet data
    data_path = '.\results\data';
    train_data = load(fullfile(data_path, 'train.mat'));
    devel_data = load(fullfile(data_path, 'devel.mat'));
    test_data = load(fullfile(data_path, 'test.mat'));
    
    train_img = train_data.data;
    train_label = train_data.label;
    devel_img = devel_data.data;
    devel_label = devel_data.label;
    test_img = test_data.data;
    test_label = test_data.label;
    
    train_num_img_ori = size(train_img, 4);
    devel_num_img = size(devel_img, 4);
    test_num_img = size(test_img, 4);
    
    if gpu_mode == 1
        trainedNet.move('gpu');
    end
    
    %% train set
    % extract fc layers from tripletnet
    img_vgg_fc_all = [];
    for i = 1 : train_num_img_ori
        fprintf('Extracting features from train samples: magin %s do %d|%d \n', num2str(magin(m)), train_num_img_ori, i);
        img_middle = train_img(:, :, :, i);
        img_middle = img_middle / 255;
        % Forward to the network
        if ~isa(img_middle, 'single')
            img_middle = single(img_middle);
        end
        if gpu_mode == 1
            img_middle = gpuArray(img_middle);
        end
        
        % train generate network
        inputs = {'input', img_middle};
        trainedNet.eval(inputs);
        img_vgg_fc = trainedNet.vars(trainedNet.getVarIndex(out_middle)).value;
        if isa(img_vgg_fc, 'gpuArray')
            img_vgg_fc = gather(img_vgg_fc);
        end
        img_vgg_fc = reshape(img_vgg_fc, [numel(img_vgg_fc) 1]);
        img_vgg_fc_all = [img_vgg_fc_all img_vgg_fc];
    end
    data = img_vgg_fc_all;
    labels = train_label;
    save(fullfile(save_path, 'train_fc_tripletnet.mat'), 'data', 'labels', '-v7.3');
    
    %% devel set
    % extract fc layers from tripletnet
    img_vgg_fc_all = [];
    for i = 1 : devel_num_img
        fprintf('Extracting features from devel samples: magin %s do %d|%d \n', num2str(magin(m)), devel_num_img, i);
        img_middle = devel_img(:, :, :, i);
        img_middle = img_middle / 255;
        % Forward to the network
        if ~isa(img_middle, 'single')
            img_middle = single(img_middle);
        end
        if gpu_mode == 1
            img_middle = gpuArray(img_middle);
        end
        
        % train generate network
        inputs = {'input', img_middle};
        trainedNet.eval(inputs);
        img_vgg_fc = trainedNet.vars(trainedNet.getVarIndex(out_middle)).value;
        if isa(img_vgg_fc, 'gpuArray')
            img_vgg_fc = gather(img_vgg_fc);
        end
        img_vgg_fc = reshape(img_vgg_fc, [numel(img_vgg_fc) 1]);
        img_vgg_fc_all = [img_vgg_fc_all img_vgg_fc];
    end
    data = img_vgg_fc_all;
    labels = devel_label;
    save(fullfile(save_path, 'devel_fc_tripletnet.mat'), 'data', 'labels', '-v7.3');
    
    %% test set
    % extract fc layers from tripletnet
    img_vgg_fc_all = [];
    for i = 1 : test_num_img
        fprintf('Extracting features from test samples: magin %s do %d|%d \n', num2str(magin(m)), test_num_img, i);
        img_middle = test_img(:, :, :, i);
        img_middle = img_middle / 255;
        % Forward to the network
        if ~isa(img_middle, 'single')
            img_middle = single(img_middle);
        end
        if gpu_mode == 1
            img_middle = gpuArray(img_middle);
        end
        
        % train generate network
        inputs = {'input', img_middle};
        trainedNet.eval(inputs);
        img_vgg_fc = trainedNet.vars(trainedNet.getVarIndex(out_middle)).value;
        if isa(img_vgg_fc, 'gpuArray')
            img_vgg_fc = gather(img_vgg_fc);
        end
        img_vgg_fc = reshape(img_vgg_fc, [numel(img_vgg_fc) 1]);
        img_vgg_fc_all = [img_vgg_fc_all img_vgg_fc];
    end
    data = img_vgg_fc_all;
    labels = test_label;
    save(fullfile(save_path, 'test_fc_tripletnet.mat'), 'data', 'labels', '-v7.3');
end


%% OULU-NPU database
magin = [0.1, 0.5, 1, 5];

for m =  2 : 2%length(magin)
    if m ~= 1
        clear trainedNet
    end
    
    %% update mode
    update_mode = 'VGG_fix'; % 'VGG_fix' or 'VGG_update'
    
    toolbox_path = '..\..\..\toolbox\matconvnet\';
    run([toolbox_path 'matlab\vl_setupnn.m']);
    
    net_path = ['.\results\Tripletnet_', update_mode, '_VideoFrame_magin_', num2str(magin(m))];
    trainedNet = load(fullfile(net_path, 'network_5.mat'));
    trainedNet = dagnn.DagNN.loadobj(trainedNet);
    layers = trainedNet.layers;
    for l = 1 : length(layers)
        layer_name = layers(l).name;
        if ~isempty(strfind(layer_name, 'same')) || ~isempty(strfind(layer_name, 'different'))% || isempty(strfind(layer_name, 'gen'))
            trainedNet.removeLayer(layer_name);
        end
    end
    trainedNet.removeLayer('triplet_loss');
    trainedNet.conserveMemory = 0;
    out_middle = 'x40';
    gpu_mode = 1;
    save_path = ['.\results\Tripletnet_', update_mode, '_VideoFrame_magin_', num2str(magin(m))];
    % load triplet data
    data_path = 'E:\Study_Lilei\GAN4Spoofing\OULU-NPU\code\overall\results\data';
    train_data = load(fullfile(data_path, 'train.mat'));
    devel_data = load(fullfile(data_path, 'devel.mat'));
    test_data = load(fullfile(data_path, 'test.mat'));
    
    train_img = train_data.data;
    train_label = train_data.label;
    devel_img = devel_data.data;
    devel_label = devel_data.label;
    test_img = test_data.data;
    test_label = test_data.label;
    
    train_num_img_ori = size(train_img, 4);
    devel_num_img = size(devel_img, 4);
    test_num_img = size(test_img, 4);
    
    if gpu_mode == 1
        trainedNet.move('gpu');
    end
    
    %% train set
    % extract fc layers from tripletnet
    img_vgg_fc_all = [];
    for i = 1 : train_num_img_ori
        fprintf('OULU-NPU Extracting features from train samples: magin %s do %d|%d \n', num2str(magin(m)), train_num_img_ori, i);
        img_middle = train_img(:, :, :, i);
        img_middle = img_middle / 255;
        % Forward to the network
        if ~isa(img_middle, 'single')
            img_middle = single(img_middle);
        end
        if gpu_mode == 1
            img_middle = gpuArray(img_middle);
        end
        
        % train generate network
        inputs = {'input', img_middle};
        trainedNet.eval(inputs);
        img_vgg_fc = trainedNet.vars(trainedNet.getVarIndex(out_middle)).value;
        if isa(img_vgg_fc, 'gpuArray')
            img_vgg_fc = gather(img_vgg_fc);
        end
        img_vgg_fc = reshape(img_vgg_fc, [numel(img_vgg_fc) 1]);
        img_vgg_fc_all = [img_vgg_fc_all img_vgg_fc];
    end
    data = img_vgg_fc_all;
    labels = train_label;
    save(fullfile(save_path, 'OULU_NPU_train_fc_tripletnet.mat'), 'data', 'labels', '-v7.3');
    
    %% devel set
    % extract fc layers from tripletnet
    img_vgg_fc_all = [];
    for i = 1 : devel_num_img
        fprintf('OULU_NPU Extracting features from devel samples: magin %s do %d|%d \n', num2str(magin(m)), devel_num_img, i);
        img_middle = devel_img(:, :, :, i);
        img_middle = img_middle / 255;
        % Forward to the network
        if ~isa(img_middle, 'single')
            img_middle = single(img_middle);
        end
        if gpu_mode == 1
            img_middle = gpuArray(img_middle);
        end
        
        % train generate network
        inputs = {'input', img_middle};
        trainedNet.eval(inputs);
        img_vgg_fc = trainedNet.vars(trainedNet.getVarIndex(out_middle)).value;
        if isa(img_vgg_fc, 'gpuArray')
            img_vgg_fc = gather(img_vgg_fc);
        end
        img_vgg_fc = reshape(img_vgg_fc, [numel(img_vgg_fc) 1]);
        img_vgg_fc_all = [img_vgg_fc_all img_vgg_fc];
    end
    data = img_vgg_fc_all;
    labels = devel_label;
    save(fullfile(save_path, 'OULU_NPU_devel_fc_tripletnet.mat'), 'data', 'labels', '-v7.3');
    
    %% test set
    % extract fc layers from tripletnet
    img_vgg_fc_all = [];
    for i = 1 : test_num_img
        fprintf('OULU_NPU Extracting features from test samples: magin %s do %d|%d \n', num2str(magin(m)), test_num_img, i);
        img_middle = test_img(:, :, :, i);
        img_middle = img_middle / 255;
        % Forward to the network
        if ~isa(img_middle, 'single')
            img_middle = single(img_middle);
        end
        if gpu_mode == 1
            img_middle = gpuArray(img_middle);
        end
        
        % train generate network
        inputs = {'input', img_middle};
        trainedNet.eval(inputs);
        img_vgg_fc = trainedNet.vars(trainedNet.getVarIndex(out_middle)).value;
        if isa(img_vgg_fc, 'gpuArray')
            img_vgg_fc = gather(img_vgg_fc);
        end
        img_vgg_fc = reshape(img_vgg_fc, [numel(img_vgg_fc) 1]);
        img_vgg_fc_all = [img_vgg_fc_all img_vgg_fc];
    end
    data = img_vgg_fc_all;
    labels = test_label;
    save(fullfile(save_path, 'OULU_NPU_test_fc_tripletnet.mat'), 'data', 'labels', '-v7.3');
end