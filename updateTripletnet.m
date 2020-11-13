function [state, net] = updateTripletnet(state, net, opts, batchSize, update_mode)

for p = 1 : numel(net.params) 
    if strcmp(update_mode, 'VGG_fix')
        name = net.params(p).name;
        if isempty(strfind(name, 'gen_'))
            continue;
        end
    end
    thisDecay = opts.weightDecay * net.params(p).weightDecay ;
    thisLR = state.learningRate * net.params(p).learningRate;
    state.momentum{p} = opts.momentum * state.momentum{p} ...
                        - thisDecay * net.params(p).value ...
                        - (1 / batchSize) * net.params(p).der;
    net.params(p).value = net.params(p).value + thisLR .* state.momentum{p};  
end