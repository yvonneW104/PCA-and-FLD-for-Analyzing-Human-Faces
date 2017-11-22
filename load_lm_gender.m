function [training_set, testing_set, training_female, training_male, testing_female, testing_male, mean_lm, mean_f, mean_m] = load_lm_gender(varargin)
    addr_f = '/Users/Yvonne/Desktop/CS276/Project/Project1/face_data/female_landmark_87';
    addr_m = '/Users/Yvonne/Desktop/CS276/Project/Project1/face_data/male_landmark_87';
    
    point = 87 * 2;
    test_size = 10;
    
    fileInfo_f = dir(fullfile(addr_f, '*.txt')); 
    fileInfo_m = dir(fullfile(addr_m, '*.txt'));

    training_set = zeros(point, numel(fileInfo_f)+numel(fileInfo_m)-2*test_size);
    testing_set = zeros(point, 2*test_size);
    training_female = zeros(point, numel(fileInfo_f)-test_size);
    training_male = zeros(point, numel(fileInfo_m)-test_size);
    testing_female = zeros(point, test_size);
    testing_male = zeros(point, test_size);
    
    for i = 1 : 75
        fileName_f = fullfile(addr_f, fileInfo_f(i).name);
        fid = fopen(fileName_f);
        datacell = textscan(fid, '%f %f');
        fclose(fid);
        for j = 1:87
           training_female(j, i) = datacell{1,1}(j); 
           training_female(j+87, i) = datacell{1,2}(j); 
        end
    end

    mean_f = mean(training_female, 2);

    for i = 1 : 10
        fileName_f = fullfile(addr_f, fileInfo_f(75+i).name);
        fid = fopen(fileName_f);
        datacell = textscan(fid, '%f %f');
        fclose(fid);
        for j = 1:87
           testing_female(j, i) = datacell{1,1}(j); 
           testing_female(j+87, i) = datacell{1,2}(j); 
        end
   end
    
    for i = 1 : 78
        fileName_m = fullfile(addr_m, fileInfo_m(i).name);
        fid = fopen(fileName_m);
        datacell = textscan(fid, '%f %f');
        fclose(fid);
        for j = 1:87
           training_male(j, i) = datacell{1,1}(j); 
           training_male(j+87, i) = datacell{1,2}(j); 
        end
    end

	mean_m = mean(training_male, 2);

   for i = 1 : 10
        fileName_m = fullfile(addr_m, fileInfo_m(78+i).name);
        fid = fopen(fileName_m);
        datacell = textscan(fid, '%f %f');
        fclose(fid);
        for j = 1:87
           testing_male(j, i) = datacell{1,1}(j); 
           testing_male(j+87, i) = datacell{1,2}(j); 
        end
   end
    
    training_set(:,1:75) = training_female;
    training_set(:,76:153) = training_male;
    testing_set(:,1:10) = testing_female;
    testing_set(:,11:20) = testing_male;
    mean_lm = mean(training_set, 2);
end