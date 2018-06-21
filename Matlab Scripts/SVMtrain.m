Data = load('TrainNew.csv');
Datacalc = [Data(1:108555/3, :); Data(108556: 108556 + (297307-108557)/2, :)];
X = Data(:, 1:40);
Y = Data(:, 41);
Xg = gpuArray(X);
Yg = gpuArray(Y);
disp('data loading done');
SVMModel = fitcsvm(Xg,Yg, 'KernelFunction', 'rbf');
disp('model training completed');
newData = load('TestNew.csv');
newX = newData(:,1:40);
newY = newData(:, 41);
newXg = gpuArray(newX);
newYg = gpuArray(newY);
[result, score] = predict(SVMModel, newXg);
disp('prediction completed');
% F-SCORE CODE
[confMat,order] = confusionmat(newY,result);
for i =1:size(confMat,1)

    recall(i)=confMat(i,i)/sum(confMat(i,:));
end

recall(isnan(recall))=[];

Recall=sum(recall)/size(confMat,1);
for i =1:size(confMat,1)

    precision(i)=confMat(i,i)/sum(confMat(:,i));
end

Precision=sum(precision)/size(confMat,1);

%%% F-score

F_score=2*Recall*Precision/(Precision+Recall); %%F_score=2*1/((1/Precision)+(1/Recall));