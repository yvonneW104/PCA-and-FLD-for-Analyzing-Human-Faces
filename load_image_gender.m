function [training_set, testing_set, training_female, training_male, testing_female, testing_male, mean_face, mean_f, mean_m] = load_image_gender(varargin)
    addr_f = '/Users/Yvonne/Desktop/CS276/Project/Project1/face_data/female_face';
    addr_m = '/Users/Yvonne/Desktop/CS276/Project/Project1/face_data/male_face';
    
    pixel = 256 * 256;
    test_size = 10;

    fileInfo_f = dir(fullfile(addr_f, '*.bmp')); % list all .bmp file content
    fileInfo_m = dir(fullfile(addr_m, '*.bmp'));

    training_set = zeros(pixel, numel(fileInfo_f)+numel(fileInfo_m)-2*test_size);
    testing_set = zeros(pixel, 2*test_size);
    training_female = zeros(pixel, numel(fileInfo_f)-test_size);
    training_male = zeros(pixel, numel(fileInfo_m)-test_size);
    testing_female = zeros(pixel, test_size);
    testing_male = zeros(pixel, test_size);
    
    for i = 1 : 75
        fileName_f = fullfile(addr_f, fileInfo_f(i).name);
        img = double(reshape(imread(fileName_f), [pixel, 1]));
        training_female(:, i) = img;
    end

    mean_f = mean(training_female, 2);

   for i = 1 : 10
        fileName_f = fullfile(addr_f, fileInfo_f(75+i).name);
        img = double(reshape(imread(fileName_f), [pixel, 1]));
        testing_female(:, i) = img;
   end
    
    for i = 1 : 78
        fileName_m = fullfile(addr_m, fileInfo_m(i).name);
        img = double(reshape(imread(fileName_m), [pixel, 1]));
        training_male(:, i) = img;
    end

	mean_m = mean(training_male, 2);

   for i = 1 : 10
        fileName_m = fullfile(addr_m, fileInfo_m(78+i).name);
        img = double(reshape(imread(fileName_m), [pixel, 1]));
        testing_male(:, i) = img;
   end
    
    training_set(:,1:75) = training_female;
    training_set(:,76:153) = training_male;
    testing_set(:,1:10) = testing_female;
    testing_set(:,11:20) = testing_male;
    mean_face = mean(training_set, 2);
end