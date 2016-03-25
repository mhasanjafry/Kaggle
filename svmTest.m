clear; clc; format long g;
%% Input
trainM = csvread('train.csv',1,1);
testM = csvread('test.csv',1,0);
train.x=trainM(:,1:end-1);%1:5000,1:end-1);
train.y=trainM(:,end);
test.x=testM(:,2:end);
test.y=csvread('sample_submission.csv',1,1);
%test.y=test.y(1:5000,:);

%% Normalization: feature-min(feature)/(max(feature)-min(feature))
% for i=1:size(test.x,2)
%     maxVal = max(max(test.x(:,i)),max(train.x(:,i)));
%     minVal = min(min(test.x(:,i)),min(train.x(:,i)));
%     test.x(:,i) = (test.x(:,i)-minVal)/(maxVal-minVal);
%     train.x(:,i) = (train.x(:,i)-minVal)/(maxVal-minVal);
% end

%% Classifier
svmStruct = svmtrain(train.y, train.x,...
        sprintf('-c %f -t %d', 4^6, 2));
[predict_label, accuracy, prob_values] = svmpredict(test.y, test.x, svmStruct);


%% Output
headers = {'ID' 'TARGET'};
outMat = [testM(:,1) predict_label];
fileID = fopen('csvout.csv','wt');
[rows, columns] = size(headers);
for index = 1:rows    
    fprintf(fileID, '%s,', headers{index,1:end});
end 
fprintf(fileID, '\n');
fclose(fileID);
dlmwrite('csvout.csv', outMat, 'precision', 10, '-append', 'delimiter', ',');