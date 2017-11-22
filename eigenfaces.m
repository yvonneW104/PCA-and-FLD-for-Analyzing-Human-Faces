close all;
clear all;

training_size = 150;
testing_size = 27;

% Question 1
% Load training & testing images
[training_face, mean_face, test_face] = load_image();

figure('Name','Mean Face of Training Data');
imshow(reshape(mean_face, [256 256]),[]);   % show mean face

% Get first 20 eigenvectors
aligned_face = training_face - repmat(mean_face, 1, training_size);
[eigen_vector, score, eigen_value] = pca(aligned_face.');
eigen_face = eigen_vector(:, 1:20); 

figure('Name','First 20 Eigen Faces');
for i = 1 : 20
    subplot(4, 5, i);
    imshow(reshape(eigen_face(:, i), [256, 256]), []);
end

% Nomalize eigen faces
normalized_eigenface = zeros(256*256, 20);   % 256^2 x 20
for  i = 1 : 20
    normalized_eigenface(:,i) = eigen_face(:,i)/sqrt(eigen_face(:,i).'*eigen_face(:,i));
end

% reconstruction error for first 20 eigenfaces
aligned_test_face = test_face - repmat(mean_face, 1, testing_size);
bi = normalized_eigenface' * aligned_test_face;
sum_biei = zeros(256*256, testing_size);
err_q1 = zeros(1, 20);
for i = 1 : 20
    sum_biei = sum_biei + normalized_eigenface(:, i) * bi(i, :);
    y_hat = repmat(mean_face, 1, testing_size) + sum_biei;
    err_q1(i) = sum(sum((test_face - y_hat).^2),2) / (256*256*27);
end
figure('Name','Total Reconstruction Error Per Pixel Over 20 Eigenfaces');
plot(err_q1, '-o');
xlabel('Number of Eigenfaces');
ylabel('Reconstruction Error for Test Images');

%Show reconstructed test face
figure('Name','Reconstructed Test Face');
for i = 1:testing_size
    subplot(5,6,i);
    imshow(reshape(y_hat(:, i),[256, 256]),[]);
end

%show original test face
figure('Name','Original Test Face');
for i = 1:testing_size
    subplot(5,6,i);
    imshow(reshape(test_face(:, i),[256, 256]),[]);
end



% Question 2
% Load training & testing landmarks
[training_lm, mean_lm, test_lm] = load_lm();

% Get first 5 eigenvectors
aligned_lm = training_lm - repmat(mean_lm, 1, training_size);
[eigen_lm, score, evalue_lm] = pca(aligned_lm.');
eigen_lm = eigen_lm(:, 1:5); 

% Nomalize eigen landmark
normalized_eigen_lm = zeros(87*2, 5);   % 256^2 x 20
for  i = 1:5
    normalized_eigen_lm(:,i) = eigen_lm(:,i)/sqrt(eigen_lm(:,i).'*eigen_lm(:,i));
end

% Display mean landmark on mean face
figure;
imshow(reshape(mean_face,[256,256]), []);
hold on
plot(mean_lm(1:87, 1), mean_lm(88:174, 1), 'r.', 'markers', 10);
hold off


% Display 5 EigenLandmark on Mean Face
eigen_lm_w_mean = zeros(87*2, 5);
for i = 1:5
    eigen_lm_w_mean(:,i) = eigen_lm(:,i) * sqrt(evalue_lm(i)) + mean_lm;
end

figure;
imshow(reshape(mean_face,[256,256]), []);
hold on
plot(eigen_lm_w_mean(1:87, 1), eigen_lm_w_mean(88:174, 1), 'r.', 'markers', 10);
plot(eigen_lm_w_mean(1:87, 2), eigen_lm_w_mean(88:174, 2), 'y.', 'markers', 10);
plot(eigen_lm_w_mean(1:87, 3), eigen_lm_w_mean(88:174, 3), 'm.', 'markers', 10);
plot(eigen_lm_w_mean(1:87, 4), eigen_lm_w_mean(88:174, 4), 'b.', 'markers', 10);
plot(eigen_lm_w_mean(1:87, 5), eigen_lm_w_mean(88:174, 5), 'g.', 'markers', 10);
hold off

% Reconstruction error for first 5 eigen landmarks
aligned_test_lm = test_lm - repmat(mean_lm, 1, testing_size);
ai = normalized_eigen_lm.' * aligned_test_lm;
sum_aiei = zeros(87*2, testing_size);
err_q2 = zeros(1, 5);
for i = 1:5
    sum_aiei = sum_aiei + normalized_eigen_lm(:,i) * ai(i,:);
    x_hat = sum_aiei + repmat(mean_lm, 1, testing_size);
    for j=1:testing_size
        err_q2(i) = err_q2(i) + norm(x_hat(:,j)-test_lm(:,j));
    end
    err_q2(i)= err_q2(i)/27;
end
figure('Name','Reconstruction Error Over 5 EigenLandMark');
plot(err_q2, '-o');
xlabel('Number of EigenLandmark');
ylabel('Reconstruction Error for Test Landmark');

figure;
for i = 1:testing_size
    subplot(5,6,i);
    warped_test_img = warpImage_new(reshape(test_face(:,i),[256,256]), ...
                reshape(x_hat(:,i),[87,2]), reshape(mean_lm,[87,2]));
    imshow(warped_test_img, []);
    hold on
    plot(x_hat(1:87,i), x_hat(88:174,i), 'r.');
    plot(test_lm(1:87,i), test_lm(88:174,i),'b.');
    hold off
end


% Question 3
% Compute First 10 EigenLandmark
[eigen_warping, score, evalue_warping] = pca(aligned_lm.');
eigen_warping = eigen_warping(:, 1:10); 

% Reconstruct landmark using 10 eigenwarping
ai_warp = eigen_warping.' * aligned_test_lm;
x_hat_warp = eigen_warping * ai_warp + repmat(mean_lm, 1, testing_size);

% Warp Training Image to Landmark
training_face_warped = zeros(256*256,training_size);
for i = 1:training_size
   training_face_warped(:,i) = reshape(warpImage_new(reshape(training_face(:,i),[256,256]), ...
                  reshape(training_lm(:,i),[87,2]), reshape(mean_lm,[87,2])),[256*256,1]);
end
mean_face_warped = mean(training_face_warped, 2);

% Compute First 10 Eigenface
aligned_face_warped = training_face_warped - repmat(mean_face_warped, 1, training_size);
[eigen_app, score, evalue_app] = pca(aligned_face_warped.');
eigen_app = eigen_app(:, 1:10); 

% Warping Test Image to Landmark
test_face_warped = zeros(256*256, testing_size);
for i = 1:testing_size
	test_face_warped(:,i) = reshape(warpImage_new(reshape(test_face(:,i),[256,256]), ...
                  reshape(test_lm(:,i),[87,2]), reshape(mean_lm,[87,2])),[256*256,1]);
end
aligned_test_face_warp = test_face_warped - repmat(mean_face_warped, 1, testing_size);

err_q3 = zeros(1, 10);

for i = 1:10
    ei = eigen_app(:, 1:i);
	bi_warped = ei' * aligned_test_face_warp;
	y_hat_warped = ei * bi_warped + repmat(mean_face_warped, 1, testing_size);
   
	reconstructed_img = zeros(256*256, testing_size);
    for j = 1:27
        reconstructed_img(:, j) = reshape(warpImage_new(reshape(y_hat_warped(:,j),[256,256]), ...
                    reshape(mean_lm,[87,2]), reshape(x_hat_warp(:,j),[87,2])), [256*256,1]);
    end
    err_q3(i) = sum(sum((test_face - reconstructed_img).^2),2) / (256*256*27);
end
figure('Name','Reconstruction Q3');
plot(err_q3, '-o');
xlabel('Number of Eigenfaces');
ylabel('Reconstruction Error for Test Images');


figure('Name','Reconstructed Test Face Q3');
for i = 1:27
    subplot(5,6,i)
    imshow(reshape(reconstructed_img(:,i),[256, 256]), []);
end

figure('Name','Original Test Face');
for i = 1:testing_size
    subplot(5,6,i);
    imshow(reshape(test_face(:, i),[256, 256]),[]);
end

% Question 4
figure;
for i = 1:20
    nor_lm = zeros(10, 1);
    nor_app = zeros(10, 1);

    for j = 1:10
        nor_lm(j, 1) = normrnd(0, sqrt(evalue_lm(j,1)));
        nor_app(j, 1) = normrnd(0, sqrt(evalue_warping(j,1)));
    end
    
    synthesized_lm = eigen_warping * nor_lm + mean_lm;
    synthesized_app = eigen_app * nor_app + mean_face_warped;

    synthesized_face = warpImage_new(reshape(synthesized_app,[256,256 ]), ...
                       reshape(mean_lm,[87,2]), reshape(synthesized_lm,[87,2]));
     subplot(4,5,i)
     imshow(synthesized_face, []);
end