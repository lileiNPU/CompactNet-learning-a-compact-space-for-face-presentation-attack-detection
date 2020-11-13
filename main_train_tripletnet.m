clc;
clear;
close all;

magin = [0.1, 0.5, 1, 5, 10];
debug = 0;

for m =  2 : 2%length(magin)
    if m ~= 1
        clear trainedNet VGG_model net_triplet
    end
    %% update mode
    update_mode = 'VGG_fix'; % 'VGG_fix' or 'VGG_update'
    
    toolbox_path = '..\..\..\toolbox\matconvnet\';
    run([toolbox_path 'matlab\vl_setupnn.m']);
    
    net_path = '.\results\Resnet';
    trainedNet = load(fullfile(net_path, 'network_30.mat'));
    trainedNet = dagnn.DagNN.loadobj(trainedNet);
    trainedNet.removeLayer('eucdisloss');
    trainedNet.removeLayer('gen_substract_1');
    
    % load pretrained VGG19 model
    model_path = '.\results\pretrainedModel';
    VGG_model = load(fullfile(model_path, 'imagenet-vgg-verydeep-19.mat'));
    VGG_model = dagnn.DagNN.fromSimpleNN(VGG_model, 'canonicalNames', true);
    VGG_model.removeLayer('prob');
    VGG_model.removeLayer('fc8');
    VGG_model.removeLayer('relu7');
    % VGG_model.removeLayer('fc7');
    % VGG_model.removeLayer('relu6');
    % VGG_model.removeLayer('fc6');
    % VGG_model.removeLayer('pool5');
    % VGG_model.removeLayer('relu5_4');
    
    VGG_model.layers(1).inputs{1} = 'map_gen_41';
    trainedNet = trainedNet.saveobj();
    VGG_model = VGG_model.saveobj();
    
    net_middle.vars = [trainedNet.vars, VGG_model.vars];
    net_middle.layers = [trainedNet.layers, VGG_model.layers];
    net_middle.params = [trainedNet.params, VGG_model.params];
    
    clear trainedNet VGG_model;
    % same and different network
    net_same = net_middle;
    net_different = net_middle;
    
    for i = 1 : length(net_same.layers)
        net_same.layers(i).name = strcat(net_same.layers(i).name, '_same');
        for j = 1 : length(net_same.layers(i).inputs)
            net_same.layers(i).inputs{j} = strcat(net_same.layers(i).inputs{j}, '_same');
        end
        for j = 1 : length(net_same.layers(i).outputs)
            net_same.layers(i).outputs{j} = strcat(net_same.layers(i).outputs{j}, '_same');
        end
    end
    for i = 1 : length(net_same.vars)
        net_same.vars(i).name = strcat(net_same.vars(i).name, '_same');
    end
    
    for i = 1 : length(net_different.layers)
        net_different.layers(i).name = strcat(net_different.layers(i).name, '_different');
        for j = 1 : length(net_different.layers(i).inputs);
            net_different.layers(i).inputs{j} = strcat(net_different.layers(i).inputs{j}, '_different');
        end
        for j = 1 : length(net_different.layers(i).outputs);
            net_different.layers(i).outputs{j} = strcat(net_different.layers(i).outputs{j}, '_different');
        end
    end
    for i = 1 : length(net_different.vars)
        net_different.vars(i).name = strcat(net_different.vars(i).name, '_different');
    end
    
    net_triplet.vars = [net_same.vars, net_middle.vars, net_different.vars];
    net_triplet.layers = [net_same.layers, net_middle.layers, net_different.layers];
    net_triplet.params = [net_same.params, net_middle.params, net_different.params];
    
    net_triplet = dagnn.DagNN.loadobj(net_triplet) ;
    out_same = 'x40_same';
    out_middle = 'x40';
    out_different = 'x40_different';
    
    triplet_input = {out_same, out_middle, out_different};
    tripletlossBlock = dagnn.TripletLoss() ;
    tripletlossBlock.magin = magin(m);
    net_triplet.addLayer('triplet_loss', tripletlossBlock, triplet_input, {'objective'});
    
    %% training parameters
    change_center_mode = 0;
    state.learningRate = 2e-5;
    state.momentum = num2cell(zeros(1, numel(net_triplet.params)));
    net_triplet.conserveMemory = 0;
    net_triplet.mode = 'normal';
    opts.weightDecay = 0.0005 ;
    opts.momentum = 0.9 ;
    epoch = 5;
    batch_size = 1;
    gpu_mode = 1;
    save_path = ['.\results\Tripletnet_', update_mode, '_magin_', num2str(magin(m))];
    if ~exist(save_path)
        mkdir(save_path);
    end
    if gpu_mode == 1
        state.momentum = cellfun(@gpuArray, state.momentum, 'UniformOutput', false);
    end
    
    % load triplet data
    data_path = '.\results\data';
    train_data = load(fullfile(data_path, 'train.mat'));
    devel_data = load(fullfile(data_path, 'devel.mat'));
    
    train_img = train_data.data;
    train_label = train_data.label;
    devel_img = devel_data.data;
    devel_label = devel_data.label;
    
    train_num_img_ori = size(train_img, 4);
    devel_num_img = size(devel_img, 4);
    
    % just for debug
    if debug == 1
        img_select_index = randperm(train_num_img_ori);
        train_img = train_img(:, :, :, img_select_index(1 : 3000));
        train_label = train_label(img_select_index(1 : 3000));
        train_num_img_ori = size(train_img, 4);
    end
    
    loss_train_all_epoch = [];
    loss_devel_all_epoch = [];
    for e = 1 : epoch
        
        if gpu_mode == 1
            net_triplet.move('gpu');
        end
        
        %% select samples for tripletloss
        img_vgg_fc_all = single([]);
        for i = 1 : train_num_img_ori
            fprintf('Select training samples for triplet loss: magin %s epoch %d|%d do %d|%d \n', num2str(magin(m)), epoch, e, train_num_img_ori, i);
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
            img_same = img_middle;
            img_different = img_middle;
            inputs = {'input', img_middle, 'input_same', img_same, 'input_different', img_different};
            net_triplet.eval(inputs);
            img_vgg_fc = net_triplet.vars(net_triplet.getVarIndex(out_middle)).value;
            if isa(img_vgg_fc, 'gpuArray')
                img_vgg_fc = gather(img_vgg_fc);
            end
            img_vgg_fc = reshape(img_vgg_fc, [numel(img_vgg_fc) 1]);
            img_vgg_fc_all = [img_vgg_fc_all img_vgg_fc];
        end
        if change_center_mode == 1
            [train_img_middle, train_img_same, train_img_different] = visualizeVGGfeatures(train_img, img_vgg_fc_all, train_label, e, save_path);
            if isempty(train_img_same)
                fprintf('The train samples are empty! \n');
                break;
            end
        else
            if e == 1
                % visualize the distribution of train data
                [train_img_middle, train_img_same, train_img_different] = visualizeVGGfeatures(train_img, img_vgg_fc_all, train_label, e, save_path);
            else
                % visualize the distribution of train data
                [~, ~, ~] = visualizeVGGfeatures(train_img, img_vgg_fc_all, train_label, e, save_path);
            end
        end
        
        %%
        train_num_img = size(train_img_middle, 4);
        img_select_index = randperm(train_num_img);
        
        loss_train_all = 0;
        loss_devel_all = 0;
        for i = 1 : batch_size : train_num_img
            batchTrainStart = min(i, train_num_img);
            batchTrainEnd = min(i + batch_size - 1, train_num_img);
            batchNum = batchTrainEnd - batchTrainStart + 1;
            img_middle = train_img_middle(:, :, :, img_select_index(batchTrainStart : batchTrainEnd));
            img_same = train_img_same(:, :, :, img_select_index(batchTrainStart : batchTrainEnd));
            img_different = train_img_different(:, :, :, img_select_index(batchTrainStart : batchTrainEnd));
            
            img_middle = img_middle / 255;
            img_same = img_same / 255;
            img_different = img_different / 255;
            
            % Forward to the network
            if ~isa(img_middle, 'single')
                img_middle = single(img_middle);
                img_same = single(img_same);
                img_different = single(img_different);
            end
            if gpu_mode == 1
                img_middle = gpuArray(img_middle);
                img_same = gpuArray(img_same);
                img_different = gpuArray(img_different);
            end
            
            % train generate network
            inputs = {'input', img_middle, 'input_same', img_same, 'input_different', img_different};
            net_triplet.eval(inputs, {'objective', 1});
            % update the parameters
            [state, net_triplet] = updateTripletnet(state, net_triplet, opts, batch_size, update_mode);
            
            loss_eucdis = net_triplet.vars(net_triplet.getVarIndex('objective')).value;
            fprintf('Pre-train resnet - magin %s epoch: %d|%d minibatch: %d|%d Loss_eucdis=%s \n', num2str(magin(m)), epoch, e, train_num_img, i, num2str(loss_eucdis));
            
            loss_train_all = loss_train_all + loss_eucdis;
            
        end
        
        loss_train_all_epoch = [loss_train_all_epoch (loss_train_all / train_num_img) * batch_size];
        figure(1);
        semilogy(1 : e, loss_train_all_epoch, 'ro-');
        
        xlabel('epoch');
        title('The cost of Tripletnet.');
        grid on;
        drawnow;
        print(1, fullfile(save_path, 'resnet.pdf'), '-dpdf');
        
        network_save = net_triplet.saveobj() ;
        save(fullfile(save_path, ['network_', num2str(e), '.mat']), '-struct', 'network_save', '-v7.3');
    end
end