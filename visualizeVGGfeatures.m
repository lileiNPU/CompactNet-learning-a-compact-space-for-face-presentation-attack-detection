function [data_middle, data_same, data_different] = visualizeVGGfeatures(train_img, fc_data, label, epoch, save_path)
% fc_data = double(train_data.fc_data);
% label = train_data.labels;
fc_data = fc_data';
[U_1, S_1] = pca(fc_data);
U_k_1 = U_1(:, 1 : 3);
fc_data_two = fc_data * U_k_1;

figure(2);
real_new = fc_data_two(label == 1, :);
fake_new = fc_data_two(label ~= 1, :);
real_img = train_img(:, :, :, label == 1);
fake_new_1 = fc_data_two(label == 2, :);
fake_new_2 = fc_data_two(label == 3, :);
fake_new_3 = fc_data_two(label == 4, :);
fake_img = train_img(:, :, :, label ~= 1);
plot3(real_new(:, 1), real_new(:, 2), real_new(:, 3), 'ro');
hold on;
plot3(fake_new_1(:, 1), fake_new_1(:, 2), fake_new_1(:, 3), 'go');
hold on;
plot3(fake_new_2(:, 1), fake_new_2(:, 2), fake_new_2(:, 3), 'bo');
hold on;
plot3(fake_new_3(:, 1), fake_new_3(:, 2), fake_new_3(:, 3), 'ko');
grid on;
legend('Real faces', 'Printed photo atatcks', 'Displayed image attacks', 'Replayed video attacks');

% ball for the data
real_ball_center = mean(real_new, 1);
fake_ball_center = mean(fake_new, 1);

% coumpute the real face that in the outside of ball
real_ball_center_matrix = repmat(real_ball_center, [size(real_new, 1), 1]);
dis_real_2_center = (real_new - real_ball_center_matrix).^2;
dis_real_2_center = sqrt(sum(dis_real_2_center, 2));
middle_index = find(dis_real_2_center == min(dis_real_2_center(:)));
% compute radius of the ball
% real
real_anchor = real_new(middle_index, :);
if middle_index ~= 1
    remainder_point = [real_new(1 : middle_index - 1, :); real_new(middle_index + 1 : end, :)];
else
    remainder_point = real_new(2 : end, :);
end
real_anchor = repmat(real_anchor, size(remainder_point, 1) , 1);
dis = (real_anchor - remainder_point).^2;
real_radius = sqrt(sum(dis, 2));
real_radius = 0.5*mean(real_radius(:));
index_outside_real = dis_real_2_center >= real_radius;
real_outside = real_img(:, :, :, index_outside_real == 1);
real_inside = real_img(:, :, :, index_outside_real == 0);

% coumpute the fake face that in the inside of ball
real_ball_center_matrix = repmat(real_ball_center, [size(fake_new, 1), 1]);
dis_fake_2_center = (fake_new - real_ball_center_matrix).^2;
dis_fake_2_center = sqrt(sum(dis_fake_2_center, 2));
index_inside_fake = dis_fake_2_center <= 15 * real_radius;
fake_outside = fake_img(:, :, :, index_inside_fake == 0);
fake_inside = fake_img(:, :, :, index_inside_fake == 1);

x_real = real_ball_center(1);
y_real = real_ball_center(2);
z_real = real_ball_center(3);
r_real = real_radius;

[x,y,z] = sphere;
mesh(x_real + r_real * x, y_real + r_real * y, z_real + r_real * z)
alpha(0.2);
colormap([0, 1, 0]);
view(215, 10); 
title(['The distribution of train set: Epoch-', num2str(epoch)]);
saveas(gcf, fullfile(save_path, ['distribution-', num2str(epoch), '.fig']));

% drawnow;
% print(2, fullfile(save_path, ['distribution-', num2str(epoch), '.pdf']), '-dpdf');
% pause(30);
close(figure(2));

% contrust triplet training set
num_real_inside = size(real_inside, 4);
num_real_outside = size(real_outside, 4);
num_fake_inside = size(fake_inside, 4);
num_fake_outside = size(fake_outside, 4);
data_middle = single([]);
data_same = single([]);
data_different = single([]);

if num_real_outside >= 1
    for i = 1 : num_real_outside
        data_middle(:, :, :, i) = real_img(:, :, :, middle_index);
        data_same(:, :, :, i) = real_outside(:, :, :, i);
        if num_fake_inside ~= 0
            different_index = randperm(num_fake_inside);
            data_different(:, :, :, i) = fake_inside(:, :, :, different_index(1));
        else
            different_index = randperm(num_fake_outside);
            data_different(:, :, :, i) = fake_outside(:, :, :, different_index(1));
        end
    end
end

if num_fake_inside >= 1
    for i = 1 : num_fake_inside
        data_middle(:, :, :, end + 1) = real_img(:, :, :, middle_index);
        if num_real_outside ~= 0
            same_index = randperm(num_real_outside);
            data_same(:, :, :, end + 1) = real_outside(:, :, :, same_index(1));
        else
            same_index = randperm(num_real_inside);
            data_same(:, :, :, end + 1) = real_inside(:, :, :, same_index(1));
        end
        data_different(:, :, :, end + 1) = fake_inside(:, :, :, i);
    end
end
