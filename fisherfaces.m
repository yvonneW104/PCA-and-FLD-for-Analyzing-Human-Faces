close all;
clear all;

d = 153;
pixel = 256*256;

% load training & testing images
[training_face, testing_face, training_face_f, training_face_m, testing_face_f, testing_face_m, mean_face, mean_face_f, mean_face_m] = load_image_gender();

B = training_face.' * training_face;
[eigen_vector, eigen_value] = eig(B);

A = zeros(pixel, d);
for i = 1:d
    Cei = training_face * eigen_vector(:,i);
    A(:,i) = eigen_value(i,i)^0.5 * Cei / norm(Cei, 2);
end

y = A.' * (mean_face_f - mean_face_m);
z = (eigen_value^2 * eigen_vector.') \ y;
w = training_face * z;

figure('Name', 'Projected points of training face');
hold on
plot(w.' * training_face_f, 'r.');
plot(w.' * training_face_m, 'b.');
hold off
legend('female','male')


figure('Name', 'Projected points of testing face');
hold on
plot(w.' * testing_face_f, 'r.');
plot(w.' * testing_face_m, 'b.');
hold off
legend('female','male')


 
% Question 6
[training_lm, testing_lm, training_lm_f, training_lm_m, testing_lm_f, testing_lm_m, mean_lm, mean_lm_f, mean_lm_m] = load_lm_gender();

aligned_lm = training_lm - repmat(mean_lm, 1, d);
[eigen_lm, score, evalue_lm] = pca(aligned_lm.');
eigen_lm = eigen_lm(:, 1:10); 

projection_train_lm_f = eigen_lm.' * training_lm_f;
projection_train_lm_m = eigen_lm.' * training_lm_m;
projection_test_lm_f = eigen_lm.' * testing_lm_f;
projection_test_lm_m = eigen_lm.' * testing_lm_m;

mf_lm = mean(projection_train_lm_f, 2);
mm_lm = mean(projection_train_lm_m, 2);

x_minus_mean_f = projection_train_lm_f - repmat(mf_lm,1,75);
x_minus_mean_m = projection_train_lm_m - repmat(mm_lm,1,78);
S_lm = x_minus_mean_f * x_minus_mean_f.' + x_minus_mean_m * x_minus_mean_m.';
w_lm = S_lm^(-1)* (mf_lm - mm_lm);

training_face_warped = zeros(256*256,d);
for i = 1:d
   training_face_warped(:,i) = reshape(warpImage_new(reshape(training_face(:,i),[256,256]), ...
                  reshape(training_lm(:,i),[87,2]), reshape(mean_lm,[87,2])),[256*256,1]);
end
mean_face_warped = mean(training_face_warped, 2);
training_face_warped_f = training_face_warped(:,1:75);
training_face_warped_m = training_face_warped(:,76:153);

testing_face_warped = zeros(256*256,20);
for i = 1:20
   testing_face_warped(:,i) = reshape(warpImage_new(reshape(testing_face(:,i),[256,256]), ...
                  reshape(testing_lm(:,i),[87,2]), reshape(mean_lm,[87,2])),[256*256,1]);
end
testing_face_warped_f = testing_face_warped(:,1:10);
testing_face_warped_m = testing_face_warped(:,11:20);

aligned_face_warped = training_face_warped - repmat(mean_face_warped, 1, 153);
[eigen_app, score, evalue_app] = pca(aligned_face_warped.');
eigen_app = eigen_app(:, 1:10); 

projection_train_face_f = eigen_app.' * training_face_warped_f;
projection_train_face_m = eigen_app.' * training_face_warped_m;
projection_test_face_f = eigen_app.' * testing_face_warped_f;
projection_test_face_m = eigen_app.' * testing_face_warped_m;

mf_face = mean(projection_train_face_f, 2);
mm_face = mean(projection_train_face_m, 2);

S_face = (projection_train_face_f - repmat(mf_face, 1, 75)) * (projection_train_face_f - repmat(mf_face, 1, 75)).' ...
    +(projection_train_face_m - repmat(mm_face, 1, 78)) * (projection_train_face_m - repmat(mm_face, 1, 78)).';
w_face = S_face^(-1) * (mf_face - mm_face);

figure('Name', 'Projected points of testing face');
hold on
plot(projection_train_lm_f.' * w_lm, projection_train_face_f.' * w_face, 'r.');
plot(projection_train_lm_m.' * w_lm, projection_train_face_m.' * w_face, 'b.');
plot(projection_test_lm_f.' * w_lm, projection_test_face_f.' * w_face, 'rx');
plot(projection_test_lm_m.' * w_lm, projection_test_face_m.' * w_face, 'bx');
hold off
legend('female','male','female','male')
xlabel('landmark projection');
ylabel('face projection');