clc;
clear;
close all;

magin = [0.1, 0.5, 1, 5];

Results = [];
for m =  1 : 4%length(magin)
    if m ~= 1
        clear trainedNet
    end
    %% update mode
    update_mode = 'VGG_fix'; % 'VGG_fix' or 'VGG_update'
    save_path = ['.\results\protocol_', update_mode];
    if ~exist(save_path)
        mkdir(save_path);
    end
    
    fea_path = ['.\results\Tripletnet_', update_mode, '_magin_', num2str(magin(m)), '\'];
    train_set = load([fea_path, 'train_fc_tripletnet.mat']);
    train_Data = train_set.data;
    train_labels = train_set.labels;
    train_labels(train_labels ~= 1) = 2;
    devel_set = load([fea_path, 'devel_fc_tripletnet.mat']);
    devel_Data = devel_set.data;
    devel_labels = devel_set.labels;
    devel_labels(devel_labels ~= 1) = 2;
    test_set = load([fea_path, 'test_fc_tripletnet.mat']);
    test_Data = test_set.data;
    test_labels = test_set.labels;
    test_labels(test_labels ~= 1) = 2;
    
    %% SVM Linear
    n_samples = 1;
    epc_range = [0.5 0.5];
    C = -6 : 1 : 16;
    d = 0;
    Model = [];
    addpath('..\..\..\toolbox\liblinear-1.96\liblinear-1.96\matlab');
    addpath('..\..\..\toolbox\epc');
   
    for i=1:numel(C)
        model{i} = train(train_labels', sparse(double(train_Data')), ...
            sprintf('-s %d -c %f', 0,  2^C(i)));
        [lbl, acc, dec] = predict(devel_labels', sparse(double(devel_Data')), model{i});
        [com.epc.dev, com.epc.eva, epc_cost] = epc(dec(devel_labels == 2), dec(devel_labels == 1),...
            dec(devel_labels == 2), dec(devel_labels == 1), n_samples, epc_range);
        EER_dev(i) = com.epc.dev.wer_apost(1) * 100;
    end
    [~, ind] = min(EER_dev);
    Model = model{ind};
    [lbl, acc, dec1] = predict(devel_labels', sparse(double(devel_Data')), Model);
    temp = num2str(magin(m));
    if ~isempty(strfind(temp, '.'))
        temp = strrep(temp, '.', 'point');
    end
    Results.dec1.(['Magin', temp]) = dec1; 
    Results.devel_labels.(['Magin', temp]) = devel_labels;
    [lbl, acc, dec2] = predict(test_labels', sparse(double(test_Data')), Model);
    Results.dec2.(['Magin', temp]) = dec2; 
    Results.test_labels.(['Magin', temp]) = test_labels;
    
    [com.epc.dev, com.epc.eva, epc_cost] = epc(dec1(devel_labels == 2, 1), dec1(devel_labels == 1, 1),...
        dec2(test_labels == 2, 1), dec2(test_labels == 1, 1), 1, [0.5 0.5]);
    HTER = com.epc.eva.hter_apri(1) * 100;
    Results.HTER.(['Magin', temp]) = HTER; 
    [com.epc.dev, com.epc.eva, epc_cost] = epc(dec1(devel_labels == 2, 1), dec1(devel_labels == 1, 1), dec1(devel_labels == 2, 1),dec1(devel_labels == 1,1), 1, [0.5 0.5]);
    EER = com.epc.dev.wer_apost(1) * 100;
    Results.EER.(['Magin', temp]) = EER; 
end
save(fullfile(save_path, 'Results.mat'), 'Results');

