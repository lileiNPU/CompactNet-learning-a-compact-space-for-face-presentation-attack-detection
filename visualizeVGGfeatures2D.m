function [real_outside_index fake_inside_index] = visualizeVGGfeatures2D(fc_data, label, epoch)
% fc_data = double(train_data.fc_data);
% label = train_data.labels;
fc_data = fc_data';
[U_1, S_1] = pca(fc_data);
U_k_1 = U_1(:, 1 : 2);
fc_data_two = fc_data * U_k_1;

figure(1);
real_new = fc_data_two(label == 1, :);
fake_new = fc_data_two(label == 2, :);
plot(real_new(:, 1), real_new(:, 2), 'bo');
hold on;
plot(fake_new(:, 1), fake_new(:, 2), 'ro');
grid on;

% ball for the data
real_ball_center = mean(real_new, 1);
fake_ball_center = mean(fake_new, 1);
% compute radius of the ball
% real
real_point = real_new(1, :);
remainder_point = real_new(2 : end, :);
real_point = repmat(real_point, size(remainder_point, 1) , 1);
dis = (real_point - remainder_point).^2;
real_radius = sqrt(sum(dis, 2));
real_radius = 0.6*mean(real_radius(:));

x_real = real_ball_center(1);
y_real = real_ball_center(2);
r_real = real_radius;

theta = 0 : 0.01 : 2 * pi;  
Circle1 = x_real + r_real * cos(theta);  
Circle2 = y_real + r_real * sin(theta);  
c = [123, 14, 52];  
plot(Circle1, Circle2, 'c', 'linewidth', 1);  
axis equal  


[x,y] = sphere;
mesh(x_real + r_real * x, y_real + r_real * y)
alpha(0.2);
colormap([0,1,0]);

% coumpute the real face that in the outside of ball
real_ball_center_matrix = repmat(real_ball_center, [size(real_new, 1), 1]);
dis_real_2_center = (real_new - real_ball_center_matrix).^2;
dis_real_2_center = sqrt(sum(dis_real_2_center, 2));
index_outside_real = dis_real_2_center >= real_radius;