clc;
clear;
close all;

toolbox_path = '..\..\..\toolbox\matconvnet\';
run([toolbox_path 'matlab\vl_setupnn.m']);

% init and pre-train resnet 
res_block = 5;
activate_mode = 'relu'; % 'sigmoid', 'tanh', 'relu'
bn_mode = 0;
network = initResnet4Generator(activate_mode, res_block, bn_mode);

%% training parameters
state.learningRate = 2e-5;
state.momentum = num2cell(zeros(1, numel(network.params)));
network.conserveMemory = 0;
network.mode = 'normal';
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
epoch = 300;
batch_size = 5;
gpu_mode = 1;
save_path = '.\results\Resnet';
if ~exist(save_path)
    mkdir(save_path);
end
if gpu_mode == 1
    state.momentum = cellfun(@gpuArray, state.momentum, 'UniformOutput', false);
end

% load triplet data
data_path = '.\results\TripletData';
train_data = load(fullfile(data_path, 'train_triplet_data.mat'));
devel_data = load(fullfile(data_path, 'devel_triplet_data.mat'));

train_img = train_data.center_data;
train_label = train_data.center_data;
devel_img = devel_data.center_data;
devel_label = devel_data.center_data;

train_num_img = size(train_img, 4);
devel_num_img = size(devel_img, 4);

loss_train_all_epoch = [];
loss_devel_all_epoch = [];
for e = 1 : epoch

    if gpu_mode == 1
        network.move('gpu');
    end
    img_select_index = randperm(train_num_img);
    
    loss_train_all = 0;
    loss_devel_all = 0;  
    for i = 1 : batch_size : train_num_img        
        batchTrainStart = min(i, train_num_img);
        batchTrainEnd = min(i + batch_size - 1, train_num_img);
        batchNum = batchTrainEnd - batchTrainStart + 1;
        img = train_img(:, :, :, img_select_index(batchTrainStart : batchTrainEnd));
        label = train_label(:, :, :, img_select_index(batchTrainStart : batchTrainEnd));
        
        img = img / 255;
        label = 2 * (label / 255) - 1;
              
        % Forward to the network
        if ~isa(img, 'single')
            img = single(img);
            label = single(label);
        end
        if gpu_mode == 1     
            img = gpuArray(img);
            label = gpuArray(label);
        end
        
        % train generate network
        inputs = {'input', img, 'label', label};
        network.mode = 'normal';
        network.eval(inputs, {'objective', 1});
        % update the parameters
        [state, network] = updateResnet(state, network, opts, batch_size);
        
        loss_eucdis = network.vars(network.getVarIndex('objective')).value;
        fprintf('Pre-train resnet -epoch: %d|%d minibatch: %d|%d Loss_eucdis=%s \n', epoch, e, train_num_img, i, num2str(loss_eucdis));
        
        loss_train_all = loss_train_all + loss_eucdis;
        
    end
    
    loss_train_all_epoch = [loss_train_all_epoch (loss_train_all / train_num_img) * batch_size];
    figure(1);
    semilogy(1 : e, loss_train_all_epoch, 'ro-');
  
    
    %% dev set
    for d = 1 : batch_size : devel_num_img 
        batchTrainStart = min(d, devel_num_img);
        batchTrainEnd = min(d + batch_size - 1, devel_num_img);
        batchNum = batchTrainEnd - batchTrainStart + 1;
        img = devel_img(:, :, :, batchTrainStart : batchTrainEnd);
        label = devel_label(:, :, :, batchTrainStart : batchTrainEnd);
        
        img = img / 255;
        label = 2 * (label / 255) - 1;
              
        % Forward to the network
        if ~isa(img, 'single')
            img = single(img);
            label = single(label);
        end
        if gpu_mode == 1     
            img = gpuArray(img);
            label = gpuArray(label);
        end
        
        % train generate network
        inputs = {'input', img, 'label', label};
        network.mode = 'normal';
        network.eval(inputs);
        
        loss_eucdis = network.vars(network.getVarIndex('objective')).value;
        fprintf('Test resnet -epoch: %d|%d minibatch: %d|%d Loss_eucdis=%s \n', epoch, e, train_num_img, d, num2str(loss_eucdis));
        
        loss_devel_all = loss_devel_all + loss_eucdis;    
    end
    
    loss_devel_all_epoch = [loss_devel_all_epoch (loss_devel_all / devel_num_img) * batch_size];
    figure(1);
    hold on;
    semilogy(1 : e, loss_devel_all_epoch, 'b*-');
    
    legend('Train Eucdistance cost', 'Devel Eucdistance cost');
    xlabel('epoch');
    title('The cost of eucdis.');
    grid on;
    drawnow;
    print(1, fullfile(save_path, 'resnet.pdf'), '-dpdf');
      
    network_save = network.saveobj() ;
    save(fullfile(save_path, ['network_', num2str(e), '.mat']), '-struct', 'network_save', '-v7.3');
end