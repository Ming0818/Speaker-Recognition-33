Data = load('Training Data.csv');
X = Data(:, 1:13);
Y = Data(:, 14);
newData = load('Testing Data.csv');
newX = newData(:,1:13);
newY = newData(:, 14);

GM = fitgmdist(X, 2);

figure
y = [zeros(1000,1);ones(1000,1)];
h = gscatter(X(:,1),X(:,2),y);
hold on
ezcontour(@(x1,x2)pdf(GMModel,[x1 x2]),get(gca,{'XLim','YLim'}))
title('{\bf Scatter Plot and Fitted Gaussian Mixture Contours}')
legend(h,'Model 0','Model1')
hold off
