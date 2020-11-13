function model = initResnet4Generator(active_mode, res_block, bn)
conv_gen_counter = 0;
map_gen_counter = 1;
relu_gen_counter = 0;
sigmoid_gen_counter = 0;
tanh_gen_counter = 0;
upsample_gen_counter = 0;
substract_gen_counter = 0;
eucdisloss_vgg_counter = 0;
scale_counter = 0;
batchNorm_gen_counter = 0;
sum_gen_counter = 0;
linear_counter = 0;
eraser_params = {};

%bn = 0;
model =  dagnn.DagNN();

%% Conv layer
convBlock = dagnn.Conv('size', [3, 3, 3, 64], 'hasBias', true);
convBlock.pad = [1 1 1 1];
convBlock.stride = 1;
conv_gen_counter = conv_gen_counter + 1;
model.addLayer(['gen_conv_gen_',num2str(conv_gen_counter)], convBlock, {'input'}, {['map_gen_', num2str(map_gen_counter + 1)]}, ...
              {['gen_filters', num2str(conv_gen_counter)], ['gen_biases', num2str(conv_gen_counter)]});
p = model.getParamIndex(model.layers(end).params) ;
params = model.layers(end).block.initParams();
[model.params(p).value] = deal(params{:});
map_gen_counter = map_gen_counter + 1;

if strcmp(active_mode, 'relu')
    % Relu²ã
    relu_gen_counter = relu_gen_counter + 1;
    reluBlock = dagnn.ReLU();
    reluBlock.leak = 0.2;
    model.addLayer(['gen_relu_gen_',num2str(relu_gen_counter)], reluBlock, {['map_gen_', num2str(map_gen_counter)]}, {['map_gen_', num2str(map_gen_counter + 1)]});
    map_gen_counter = map_gen_counter + 1;
end
if strcmp(active_mode, 'sigmoid')
    % Sigmoid²ã
    sigmoid_gen_counter = sigmoid_gen_counter + 1;
    sigmoidBlock = dagnn.Sigmoid();
    model.addLayer(['gen_sigmoid_gen_',num2str(sigmoid_gen_counter)], sigmoidBlock, {['map_gen_', num2str(map_gen_counter)]}, {['map_gen_', num2str(map_gen_counter + 1)]});
    map_gen_counter = map_gen_counter + 1;
end
if strcmp(active_mode, 'tanh')
    % Tanh²ã
    tanh_gen_counter = tanh_gen_counter + 1;
    tanhBlock = dagnn.Tanh();
    model.addLayer(['gen_tanh_gen_',num2str(tanh_gen_counter)], tanhBlock, {['map_gen_', num2str(map_gen_counter)]}, {['map_gen_', num2str(map_gen_counter + 1)]});
    map_gen_counter = map_gen_counter + 1;
end
 
%% Resblock
for r = 1 : res_block
    res_input = ['map_gen_', num2str(map_gen_counter)];
   
    % conv layer
    convBlock = dagnn.Conv('size', [3, 3, 64, 64], 'hasBias', true);
    convBlock.pad = [1 1 1 1];
    convBlock.stride = 1;
    conv_gen_counter = conv_gen_counter + 1;
    model.addLayer(['gen_conv_gen_',num2str(conv_gen_counter)], convBlock, {['map_gen_', num2str(map_gen_counter)]}, {['map_gen_', num2str(map_gen_counter + 1)]}, ...
                  {['gen_filters', num2str(conv_gen_counter)], ['gen_biases', num2str(conv_gen_counter)]});
    p = model.getParamIndex(model.layers(end).params) ;
    params = model.layers(end).block.initParams();
    [model.params(p).value] = deal(params{:});
    map_gen_counter = map_gen_counter + 1; 
    
    if bn == 1
        % batchNorm Layer
        ndim = size(params{1}, 4);
        batchNorm_gen_counter = batchNorm_gen_counter + 1;
        batchnormBlock = dagnn.BatchNorm();
        batchnormBlock.numChannels = ndim;
        params = initParams(batchnormBlock);
        model.addLayer(['gen_batchNorm_gen_',num2str(batchNorm_gen_counter)], batchnormBlock, {['map_gen_', num2str(map_gen_counter)]}, {['map_gen_', num2str(map_gen_counter + 1)]}, ...
            {['gen_norm_weights', num2str(batchNorm_gen_counter)], ['gen_norm_biases', num2str(batchNorm_gen_counter)], ['gen_trainMethod', num2str(batchNorm_gen_counter)]});
        p = model.getParamIndex(model.layers(end).params) ;
        [model.params(p).value] = deal(params{:});
        map_gen_counter = map_gen_counter + 1;
    end
    
    if strcmp(active_mode, 'relu')
        % Relu²ã
        relu_gen_counter = relu_gen_counter + 1;
        reluBlock = dagnn.ReLU();
        reluBlock.leak = 0.2;
        model.addLayer(['gen_relu_gen_',num2str(relu_gen_counter)], reluBlock, {['map_gen_', num2str(map_gen_counter)]}, {['map_gen_', num2str(map_gen_counter + 1)]});
        map_gen_counter = map_gen_counter + 1;
    end
    if strcmp(active_mode, 'sigmoid')
        % Sigmoid²ã
        sigmoid_gen_counter = sigmoid_gen_counter + 1;
        sigmoidBlock = dagnn.Sigmoid();
        model.addLayer(['gen_sigmoid_gen_',num2str(sigmoid_gen_counter)], sigmoidBlock, {['map_gen_', num2str(map_gen_counter)]}, {['map_gen_', num2str(map_gen_counter + 1)]});
        map_gen_counter = map_gen_counter + 1;
    end
    if strcmp(active_mode, 'tanh')
        % Tanh²ã
        tanh_gen_counter = tanh_gen_counter + 1;
        tanhBlock = dagnn.Tanh();
        model.addLayer(['gen_tanh_gen_',num2str(tanh_gen_counter)], tanhBlock, {['map_gen_', num2str(map_gen_counter)]}, {['map_gen_', num2str(map_gen_counter + 1)]});
        map_gen_counter = map_gen_counter + 1;
    end
    
    % conv layer
    convBlock = dagnn.Conv('size', [3, 3, 64, 64], 'hasBias', true);
    convBlock.pad = [1 1 1 1];
    convBlock.stride = 1;
    conv_gen_counter = conv_gen_counter + 1;
    model.addLayer(['gen_conv_gen_',num2str(conv_gen_counter)], convBlock, {['map_gen_', num2str(map_gen_counter)]}, {['map_gen_', num2str(map_gen_counter + 1)]}, ...
                  {['gen_filters', num2str(conv_gen_counter)], ['gen_biases', num2str(conv_gen_counter)]});
    p = model.getParamIndex(model.layers(end).params) ;
    params = model.layers(end).block.initParams();
    [model.params(p).value] = deal(params{:});
    map_gen_counter = map_gen_counter + 1; 
    
    if bn == 1
        % batchNorm Layer
        ndim = size(params{1}, 4);
        batchNorm_gen_counter = batchNorm_gen_counter + 1;
        batchnormBlock = dagnn.BatchNorm();
        batchnormBlock.numChannels = ndim;
        params = initParams(batchnormBlock);
        model.addLayer(['gen_batchNorm_gen_',num2str(batchNorm_gen_counter)], batchnormBlock, {['map_gen_', num2str(map_gen_counter)]}, {['map_gen_', num2str(map_gen_counter + 1)]}, ...
            {['gen_norm_weights', num2str(batchNorm_gen_counter)], ['gen_norm_biases', num2str(batchNorm_gen_counter)], ['gen_trainMethod', num2str(batchNorm_gen_counter)]});
        p = model.getParamIndex(model.layers(end).params) ;
        [model.params(p).value] = deal(params{:});
        map_gen_counter = map_gen_counter + 1;
    end
    
    res_output = ['map_gen_', num2str(map_gen_counter)];
    
    % sum layer
    sum_gen_counter = sum_gen_counter + 1;
    sum_input = {res_input, res_output};
    sumBlock = dagnn.Sum();
    model.addLayer(['gen_sum_gen_', num2str(sum_gen_counter)], sumBlock, sum_input, {['map_gen_', num2str(map_gen_counter + 1)]});
    map_gen_counter = map_gen_counter + 1;
end

%% conv layer
convBlock = dagnn.Conv('size', [3, 3, 64, 64], 'hasBias', true);
convBlock.pad = [1 1 1 1];
convBlock.stride = 1;
conv_gen_counter = conv_gen_counter + 1;
model.addLayer(['gen_conv_gen_',num2str(conv_gen_counter)], convBlock, {['map_gen_', num2str(map_gen_counter)]}, {['map_gen_', num2str(map_gen_counter + 1)]}, ...
    {['gen_filters', num2str(conv_gen_counter)], ['gen_biases', num2str(conv_gen_counter)]});
p = model.getParamIndex(model.layers(end).params) ;
params = model.layers(end).block.initParams();
[model.params(p).value] = deal(params{:});
map_gen_counter = map_gen_counter + 1;

if bn == 1
    % batchNorm Layer
    ndim = size(params{1}, 4);
    batchNorm_gen_counter = batchNorm_gen_counter + 1;
    batchnormBlock = dagnn.BatchNorm();
    batchnormBlock.numChannels = ndim;
    params = initParams(batchnormBlock);
    model.addLayer(['gen_batchNorm_gen_',num2str(batchNorm_gen_counter)], batchnormBlock, {['map_gen_', num2str(map_gen_counter)]}, {['map_gen_', num2str(map_gen_counter + 1)]}, ...
        {['gen_norm_weights', num2str(batchNorm_gen_counter)], ['gen_norm_biases', num2str(batchNorm_gen_counter)], ['gen_trainMethod', num2str(batchNorm_gen_counter)]});
    p = model.getParamIndex(model.layers(end).params) ;
    [model.params(p).value] = deal(params{:});
    map_gen_counter = map_gen_counter + 1;
end

% sum layer
sum_gen_counter = sum_gen_counter + 1;
sum_input = {'map_gen_3', ['map_gen_', num2str(map_gen_counter)]};
sumBlock = dagnn.Sum();
model.addLayer(['gen_sum_gen_', num2str(sum_gen_counter)], sumBlock, sum_input, {['map_gen_', num2str(map_gen_counter + 1)]});
map_gen_counter = map_gen_counter + 1;


%% conv layer
convBlock = dagnn.Conv('size', [3, 3, 64, 128], 'hasBias', true);
convBlock.pad = [1 1 1 1];
convBlock.stride = 1;
conv_gen_counter = conv_gen_counter + 1;
model.addLayer(['gen_conv_gen_',num2str(conv_gen_counter)], convBlock, {['map_gen_', num2str(map_gen_counter)]}, {['map_gen_', num2str(map_gen_counter + 1)]}, ...
    {['gen_filters', num2str(conv_gen_counter)], ['gen_biases', num2str(conv_gen_counter)]});
p = model.getParamIndex(model.layers(end).params) ;
params = model.layers(end).block.initParams();
[model.params(p).value] = deal(params{:});
map_gen_counter = map_gen_counter + 1;

if strcmp(active_mode, 'relu')
    % Relu²ã
    relu_gen_counter = relu_gen_counter + 1;
    reluBlock = dagnn.ReLU();
    reluBlock.leak = 0.2;
    model.addLayer(['gen_relu_gen_',num2str(relu_gen_counter)], reluBlock, {['map_gen_', num2str(map_gen_counter)]}, {['map_gen_', num2str(map_gen_counter + 1)]});
    map_gen_counter = map_gen_counter + 1;
end
if strcmp(active_mode, 'sigmoid')
    % Sigmoid²ã
    sigmoid_gen_counter = sigmoid_gen_counter + 1;
    sigmoidBlock = dagnn.Sigmoid();
    model.addLayer(['gen_sigmoid_gen_',num2str(sigmoid_gen_counter)], sigmoidBlock, {['map_gen_', num2str(map_gen_counter)]}, {['map_gen_', num2str(map_gen_counter + 1)]});
    map_gen_counter = map_gen_counter + 1;
end
if strcmp(active_mode, 'tanh')
    % Tanh²ã
    tanh_gen_counter = tanh_gen_counter + 1;
    tanhBlock = dagnn.Tanh();
    model.addLayer(['gen_tanh_gen_',num2str(tanh_gen_counter)], tanhBlock, {['map_gen_', num2str(map_gen_counter)]}, {['map_gen_', num2str(map_gen_counter + 1)]});
    map_gen_counter = map_gen_counter + 1;
end
 
%% conv layer
convBlock = dagnn.Conv('size', [3, 3, 128, 128], 'hasBias', true);
convBlock.pad = [1 1 1 1];
convBlock.stride = 1;
conv_gen_counter = conv_gen_counter + 1;
model.addLayer(['gen_conv_gen_',num2str(conv_gen_counter)], convBlock, {['map_gen_', num2str(map_gen_counter)]}, {['map_gen_', num2str(map_gen_counter + 1)]}, ...
    {['gen_filters', num2str(conv_gen_counter)], ['gen_biases', num2str(conv_gen_counter)]});
p = model.getParamIndex(model.layers(end).params) ;
params = model.layers(end).block.initParams();
[model.params(p).value] = deal(params{:});
map_gen_counter = map_gen_counter + 1;

if strcmp(active_mode, 'relu')
    % Relu²ã
    relu_gen_counter = relu_gen_counter + 1;
    reluBlock = dagnn.ReLU();
    reluBlock.leak = 0.2;
    model.addLayer(['gen_relu_gen_',num2str(relu_gen_counter)], reluBlock, {['map_gen_', num2str(map_gen_counter)]}, {['map_gen_', num2str(map_gen_counter + 1)]});
    map_gen_counter = map_gen_counter + 1;
end
if strcmp(active_mode, 'sigmoid')
    % Sigmoid²ã
    sigmoid_gen_counter = sigmoid_gen_counter + 1;
    sigmoidBlock = dagnn.Sigmoid();
    model.addLayer(['gen_sigmoid_gen_',num2str(sigmoid_gen_counter)], sigmoidBlock, {['map_gen_', num2str(map_gen_counter)]}, {['map_gen_', num2str(map_gen_counter + 1)]});
    map_gen_counter = map_gen_counter + 1;
end
if strcmp(active_mode, 'tanh')
    % Tanh²ã
    tanh_gen_counter = tanh_gen_counter + 1;
    tanhBlock = dagnn.Tanh();
    model.addLayer(['gen_tanh_gen_',num2str(tanh_gen_counter)], tanhBlock, {['map_gen_', num2str(map_gen_counter)]}, {['map_gen_', num2str(map_gen_counter + 1)]});
    map_gen_counter = map_gen_counter + 1;
end

%% conv layer
convBlock = dagnn.Conv('size', [3, 3, 128, 3], 'hasBias', true);
convBlock.pad = [1 1 1 1];
convBlock.stride = 1;
conv_gen_counter = conv_gen_counter + 1;
model.addLayer(['gen_conv_gen_',num2str(conv_gen_counter)], convBlock, {['map_gen_', num2str(map_gen_counter)]}, {['map_gen_', num2str(map_gen_counter + 1)]}, ...
    {['gen_filters', num2str(conv_gen_counter)], ['gen_biases', num2str(conv_gen_counter)]});
p = model.getParamIndex(model.layers(end).params) ;
params = model.layers(end).block.initParams();
[model.params(p).value] = deal(params{:});
map_gen_counter = map_gen_counter + 1;

%% substract layer
substract_gen_counter = substract_gen_counter + 1;
substract_input = {['map_gen_', num2str(map_gen_counter)], 'label'};
substractBlock = dagnn.Substract();
model.addLayer(['gen_substract_', num2str(substract_gen_counter)], substractBlock, substract_input, {['map_gen_', num2str(map_gen_counter + 1)]});
map_gen_counter = map_gen_counter + 1;

%% loss layer
lossBlock = dagnn.LossEucDis();
model.addLayer('eucdisloss', lossBlock, {['map_gen_', num2str(map_gen_counter)]}, {'objective'});

