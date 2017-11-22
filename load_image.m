function [training_face, mean_face, test_face] = load_image(varargin)
    addr = '/Users/Yvonne/Desktop/CS276/Project/Project1/face_data/face';
    fileInfo = dir(fullfile(addr, '*.bmp')); % list all .bmp file content
    
    training_size = 150;
    testing_size = 27;
    pixel = 256 * 256;
    
    % load training data set
    training_face = zeros(pixel, training_size);    % 256^2 x 150
    for i = 1 : training_size
        fileName = fullfile(addr, fileInfo(i).name);
        file = double(reshape(imread(fileName), [pixel, 1]));
        training_face(:, i) = file;
    end
    
    % calculate the mean face
    mean_face = mean(training_face, 2);
    
    % load testing data set
    test_face = zeros(pixel, testing_size);    % 256^2 x 27
    for i = 1 : testing_size
        fileName = fullfile(addr, fileInfo(training_size+i).name);
        file = double(reshape(imread(fileName), [pixel, 1]));
        test_face(:, i) = file;
    end