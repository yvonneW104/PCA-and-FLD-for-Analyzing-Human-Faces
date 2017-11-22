function [training_lm, mean_lm, test_lm] = load_lm(varargin)
    addr = '/Users/Yvonne/Desktop/CS276/Project/Project1/face_data/landmark_87';
    fileInfo = dir(fullfile(addr, '*.dat')); % list all .bmp file content

    training_size = 150;
    testing_size = 27;
    point = 87 * 2;
    
    % load training landmark
    training_lm = zeros(point, training_size);    % 87*2 x 150
    for i = 1 : training_size
        fileName = fullfile(addr, fileInfo(i).name);
    	file = importdata(fileName, ' ', 1);
    	training_lm(:, i) = reshape(file.data, [point, 1]);
    end
    
    % calculate the mean face
    mean_lm = mean(training_lm, 2);

    % load testing landmark
    test_lm = zeros(point, testing_size);    % 87*2 x 27
    for i = 1 : testing_size
        fileName = fullfile(addr, fileInfo(training_size+i).name);
    	file = importdata(fileName, ' ', 1);
    	test_lm(:, i) = reshape(file.data, [point, 1]);
    end