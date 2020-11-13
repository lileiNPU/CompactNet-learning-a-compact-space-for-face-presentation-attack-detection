clc;
clear;
close all;

fig_path  = '.\results\Tripletnet_VGG_fix_VideoFrame_magin_0.5';

%% Original data distribution
fig_original = open(fullfile(fig_path, 'distribution-1.fig'));
save_path = fullfile(fig_path, 'Original\');
mkdir(save_path);

grid on;
counter = 0;
for up = 10 : 10 : 50
    for i = 10 : 2 : 170
        view(i, up); % i是角度， 20为仰视角
        counter = counter + 1;
        saveas(gcf, fullfile(save_path, [num2str(counter), '.jpg']));
        pause(0.02);
    end
end

% generate gif
files = dir(fullfile(save_path, '*.jpg'));
gifName = 'OriginalDataDistribution.gif';
for v = 1 : length(files)
    img = imread(fullfile(save_path, [num2str(v), '.jpg']));
    img = imresize(img, 0.5);
     [I, map] = rgb2ind(img, 256);
     if v == 1;
         imwrite(I, map, [save_path, gifName], 'gif', 'Loopcount', inf, 'DelayTime', 0.1);%loopcount只是在i==1的时候才有用
     else
         imwrite(I, map, [save_path, gifName], 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
     end        
end

%% Learned data distribution
fig_original = open(fullfile(fig_path, 'distribution-5.fig'));
save_path = fullfile(fig_path, 'Learned\');
mkdir(save_path);

grid on;
counter = 0;
for up = 10 : 10 : 50
    for i = 10 : 2 : 170
        view(i, up); % i是角度， 20为仰视角
        counter = counter + 1;
        saveas(gcf, fullfile(save_path, [num2str(counter), '.jpg']));
        pause(0.02);
    end
end

% generate gif
files = dir(fullfile(save_path, '*.jpg'));
gifName = 'LearnedDataDistribution.gif';
for v = 1 : length(files)
    img = imread(fullfile(save_path, [num2str(v), '.jpg']));
    img = imresize(img, 0.5);
     [I, map] = rgb2ind(img, 256);
     if v == 1;
         imwrite(I, map, [save_path, gifName], 'gif', 'Loopcount', inf, 'DelayTime', 0.1);%loopcount只是在i==1的时候才有用
     else
         imwrite(I, map, [save_path, gifName], 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
     end        
end