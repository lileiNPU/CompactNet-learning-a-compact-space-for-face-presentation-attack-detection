clc;
clear;
close all;

img_show_num = 9;

magin = [0.1, 0.5, 1, 5];

for m = 2 : 2%length(magin)
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
    out_middle = 'map_gen_41';
    gpu_mode = 1;
    save_path = ['.\results\Tripletnet_', update_mode, '_VideoFrame_magin_', num2str(magin(m))];
    % load triplet data
    data_path = '.\results\data';
    train_data = load(fullfile(data_path, 'train_VideoFrame.mat'));
    
    train_img = train_data.data;
    train_label = train_data.label;
    
    train_num_img_ori = size(train_img, 4);
    
    if gpu_mode == 1
        trainedNet.move('gpu');
    end
    
    img_select_index = randperm(train_num_img_ori);
    
    %% real faces
    img_RGB_space = [];
    img_color_liked_space = [];
    img_counter = 0;
    for i = 1 : train_num_img_ori        
        img_label = train_label(img_select_index(i));
        if img_label == 1
            if img_counter < img_show_num
                img_counter = img_counter + 1;
                img_middle = train_img(:, :, :, img_select_index(i));
                temp = img_middle;
                img_RGB_space(:, :, :, img_counter) = imresize(temp, [240 320]);
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
                color_liked_space = trainedNet.vars(trainedNet.getVarIndex(out_middle)).value;
                if isa(color_liked_space, 'gpuArray')
                    color_liked_space = gather(color_liked_space);
                end
                color_liked_space = (color_liked_space - min(color_liked_space(:))) / (max(color_liked_space(:)) - min(color_liked_space(:)));
                color_liked_space = uint8(255 * double(color_liked_space));
                temp = color_liked_space;
                temp = imresize(temp, [240 320]);
                img_RGB_space(:, :, :, img_counter) = temp;
            end
        end
    end
        
    %% Visualize the RGB space and color_liked space
    temp = sqrt(img_show_num);
    col_put = temp; %the number of image in each col
    row_put = temp; %the number of image in each row
    matrix_visualize =  255 * ones(size(img_RGB_space, 1) * row_put + row_put + 1, size(img_RGB_space, 2) * col_put + col_put + 1, size(img_RGB_space, 3));
    counter = 0;
    for c = 1 : img_show_num
        counter = counter + 1;
        counter_temp = counter / col_put;
        if rem(counter_temp, 1) ~= 0
            i1 = floor(counter_temp) + 1;
        else
            i1 = counter_temp;
        end
        counter_temp = mod(counter, col_put);
        if counter_temp ~= 0
            j1 = counter_temp;
        else
            j1 = col_put;
        end
        gray_result = img_RGB_space(:, :, :, c);
        [gray_result_row, gray_result_col, gray_result_channel] = size(gray_result);
        visualization((i1 - 1) * gray_result_row + i1 + 1 : i1 * gray_result_row + i1, ...
                      (j1 - 1) * gray_result_col + j1 + 1 : j1 * gray_result_col + j1, 1 : gray_result_channel) = gray_result;
        visualization = uint8(visualization);
    end 
    figure(1);
    imshow(visualization);
    %title('Real faces in generated color-liked space');
      
    %% fake faces
    img_RGB_space = [];
    img_color_liked_space = [];
    img_counter = 0;
    for i = 1 : train_num_img_ori        
        img_label = train_label(img_select_index(i));
        if img_label ~= 1
            if img_counter < img_show_num
                img_counter = img_counter + 1;
                img_middle = train_img(:, :, :, img_select_index(i));
                temp = img_middle;
                img_RGB_space(:, :, :, img_counter) = imresize(temp, [240 320]);
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
                color_liked_space = trainedNet.vars(trainedNet.getVarIndex(out_middle)).value;
                if isa(color_liked_space, 'gpuArray')
                    color_liked_space = gather(color_liked_space);
                end
                color_liked_space = (color_liked_space - min(color_liked_space(:))) / (max(color_liked_space(:)) - min(color_liked_space(:)));
                color_liked_space = uint8(255 * double(color_liked_space));
                temp = color_liked_space;
                temp = imresize(temp, [240 320]);
                img_RGB_space(:, :, :, img_counter) = temp;
            end
        end
    end
    
    %% Visualize the RGB space and color_liked space
    temp = sqrt(img_show_num);
    col_put = temp; %the number of image in each col
    row_put = temp; %the number of image in each row
    matrix_visualize =  255 * ones(size(img_RGB_space, 1) * row_put + row_put + 1, size(img_RGB_space, 2) * col_put + col_put + 1, size(img_RGB_space, 3));
    counter = 0;
    for c = 1 : img_show_num
        counter = counter + 1;
        counter_temp = counter / col_put;
        if rem(counter_temp, 1) ~= 0
            i1 = floor(counter_temp) + 1;
        else
            i1 = counter_temp;
        end
        counter_temp = mod(counter, col_put);
        if counter_temp ~= 0
            j1 = counter_temp;
        else
            j1 = col_put;
        end
        gray_result = img_RGB_space(:, :, :, c);
        [gray_result_row, gray_result_col, gray_result_channel] = size(gray_result);
        visualization((i1 - 1) * gray_result_row + i1 + 1 : i1 * gray_result_row + i1, ...
                      (j1 - 1) * gray_result_col + j1 + 1 : j1 * gray_result_col + j1, 1 : gray_result_channel) = gray_result;
        visualization = uint8(visualization);
    end 
    figure(2);
    imshow(visualization);
    %title('Fake faces in generated color-liked space');
    
end

    
    
