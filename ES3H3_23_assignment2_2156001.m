%% ES3H3 Assignment #2 Submission
% Alvin Karanja, 2023/2024
% u2156001

%% Qusetion 1a (Linear Regression - One Predictor Variable)
clear; close all;
load('Question1a.mat');

% Fit data to model y = ax + b.

% Extend input to fit polynomial -> [x 1].
A = [x ones(size(x,1),1)];

% Perform Linear Regression: Aw = y.
% Store co-efficients in vector w -> [a b]'.
w = A \ y;

% Form predictor variable vector.
y_hat = A * w;

% Plot data.
figure; 
hold on;
title('Question 1a');

% Original data - blue points.
scatter(x, y, 'b*');

% Fitted model - red points.
scatter(x, y_hat, 'r*');

% Annotate co-efficient values.
a_str = strcat('a value:',' ',num2str(w(1)));
b_str = strcat('b value:',' ', num2str(w(2)));
str = {'Linear Regression Co-efficients', a_str, b_str};

dimensions = [0.5 0.5 0.3 0.3];
annotation('textbox', dimensions, 'String', str, 'FitBoxToText', 'on');

% Legend
legend('Original Points', 'Modeled Points');

hold off;

%% Question 1b
clear; close all;
load('Question1b.mat');

% Model Choice: Second Degree Polynomial (y = ax^2 + bx + c)
% Justification: Data contains Parabola.

% Extend input to fit polynomial -> [x^2 x 1].
A = [x.^2, x, ones(size(x,1),1)];

% Perform Linear Regression: Aw = y.
% Store co-efficients in vector w -> [a b c]'.
w = A \ y;

% Form predictor variable vector.
y_hat = A * w;

% Plot data.
figure; 
hold on;
title('Question 1b');

% Original data - blue points.
scatter(x, y, 'b*');

% Fitted model - red points.
scatter(x, y_hat, 'r*');

% Annotate co-efficient values.
a_str = strcat('a value:',' ',num2str(w(1)));
b_str = strcat('b value:',' ', num2str(w(2)));
c_str = strcat('c value:',' ', num2str(w(3)));
str = {'Linear Regression Co-efficients', a_str, b_str, c_str};

dimensions = [0.4 0.2 0.3 0.3];
annotation('textbox', dimensions, 'String', str, 'FitBoxToText', 'on');

% Legend
legend('Original Points', 'Modeled Points');

hold off;

%% Question 2 (Linear Regression - Two Predictor Variables)
clear; close all;
load('Question2.mat');

% Fit data to model:
% z = ax^2 + bx + cy^2 dy + exy + f

% Extend input to fit model.
% To perform x*y use diagonal of y.
A = [x.^2, x, y.^2, y, (x'*diag(y))', ones(size(x,1),1) ];

% Perform Linear Regression: Aw = z.
% Store co-efficients in vector w -> [a b ... f]'.
w = A \ z;

% Form predictor variable vector.
y_hat = A * w;

% Plot data.

% Original data - blue points.
scatter3(x, y, z, 'b*');

hold on;

% Fitted model - red points.
scatter3(x, y, y_hat, 'r*');

% Title
title('Question 2.b: Comparison of original data against model');
% Legend
legend('Original Points', 'Modeled Points');

hold off;

% Part c. Surface plot of model function against original data.
figure; 

% Model function (Vectorised).
f = @(x, y) (x.^2).*w(1) + x.*w(2) + (y.^2).*w(3) + y.*w(4) + ...
    (x.*y).*w(5) + w(6)*1;

% Plot model function as surface, use default range [-5 5] 
% as it corresponds with the range of the original data. 
fsurf(f);

hold on; 

% Original data - blue points.
scatter3(x, y, z, 'b*');

% Title
title('Question 2.c: Function surface plot against original data');
% Legend
legend('Function', 'Original Points');

hold off;

%% Question 3a (Gradient Descent)
clear; close all;

% Function to be minimised:
% y = 10sin(x) + (x+2)^2 + 10

% Gradient: 
% g = 10cos(x) + 2(x+2).

% Set Up - Use arbitrary range [-5 5].
xValues = -5:0.1:5;
yValues = 10 * sin(xValues) + (xValues + 2).^ 2 + 10;

% Define descent parameters.
learningRate = 0.1;
initialValue = 2;
iteratioinCounter = 1;

% Threshold is an infintesimal value.
threshold = 1e-10;

% Prepare plotting axes.
figure;
dimensions = [0.6 0.0 0.3 0.3];
plot(xValues, yValues);

% Load initial values
x_k = initialValue;
y_k = 10 * sin(x_k) + (x_k + 2).^ 2 + 10;
g = 10 * cos(x_k) + 2 * (x_k+2);

% Perfrom Gradient Descent whilst gradient value is 
% above threshold (1e-10).

% Learning annotation.
txt = annotation('textbox', dimensions, 'String', 'learning ...', ...
        'FitBoxToText', 'on');

while abs(g) > threshold

    % Plot title for iteration.
    title(strcat('Question 3.a: Gradient Descent iteration: ', ...
        num2str(iteratioinCounter)));

    hold on;

    % Plot current iteration.
    plot(x_k, y_k, 'r*', 'MarkerSize', 8);

    pause(0.5);

    % Calculate gradient.
    g = 10 * cos(x_k) + 2 * (x_k+2);

    % Compute next iteration of descent.
    % x_k+1 = x_k - ng.
    x_k = x_k - learningRate * g;
    y_k = 10 * sin(x_k) + (x_k + 2).^ 2 + 10;
    iteratioinCounter = iteratioinCounter + 1;

    hold off;

    % All plots included in axes to show model 'learning' over time.
end

% When threshold reached notify complteted descent.
delete(txt);

hold on;
message = 'Gradient Descent complete !';
finalVal = strcat('Minimised value of x:', num2str(x_k));
str = {message, finalVal};

% Text box.
annotation('textbox', dimensions, 'String', str, 'FitBoxToText', 'on');
% Plot final value.
plot(x_k, y_k, 'g*', 'MarkerSize', 16, 'LineWidth', 2);

hold off;

%% Question 3b (Newton's Method)
clear; close all;

% Function to be minimised:
% y = 10sin(x) + (x+2)^2 + 10

% Gradient: 
% g = 10cos(x) + 2(x+2).

% Second Derivative:
% h = -10sin(x) + 2.

% Set Up - Use arbitrary range [-5 5].
xValues = -5:0.1:5;
yValues = 10 * sin(xValues) + (xValues + 2).^ 2 + 10;

% Define parameters.
initialValue = -1;
iteratioinCounter = 1;
% Threshold is an infintesimal value.
threshold = 1e-10;

% Prepare plotting axes.
figure;
dimensions = [0.6 0.0 0.3 0.3];
plot(xValues, yValues);

% Load initial values
x_k = initialValue;
y_k = 10 * sin(x_k) + (x_k + 2).^ 2 + 10;
g = 10 * cos(x_k) + 2 * (x_k+2);
h = -10 * sin(x_k) + 2;

% Perfrom Newtons Method whilst gradient value is 
% above threshold (1e-10).

% Learning annotation.
txt = annotation('textbox', dimensions, 'String', 'learning ...', ...
        'FitBoxToText', 'on');

while abs(g) > threshold

    % Plot title for iteration.
    title(strcat('Question 3.b: Newtons Method iteration: ', ...
        num2str(iteratioinCounter)));

    hold on;

    % Plot current iteration.
    plot(x_k, y_k, 'r*', 'MarkerSize', 8);

    pause(0.5);

    % Calculate gradient.
    g = 10 * cos(x_k) + 2 * (x_k+2);

    % Calculate second derivative.
    h = -10 * sin(x_k) + 2;

    % Compute next iteration.
    % x_k+1 = x_k - inv(h) * g.
    % Note h^(-1) is used instead of inv(h) to avoid warning, they are
    % equivalent in operation.
    x_k = x_k - h^(-1) * g;

    y_k = 10 * sin(x_k) + (x_k + 2).^ 2 + 10;
    iteratioinCounter = iteratioinCounter + 1;

    hold off;

    % All plots included in axes to show model 'learning' over time.
end

% When threshold reached notify compltete.
delete(txt);

hold on;
message = 'Newtons Method complete !';
finalVal = strcat('Minimised value of x:', num2str(x_k));
str = {message, finalVal};

% Text box.
annotation('textbox', dimensions, 'String', str, 'FitBoxToText', 'on');
% Plot final value.
plot(x_k, y_k, 'g*', 'MarkerSize', 16, 'LineWidth', 2);

hold off;

%% Question 3c (Simulated Annealing)
clear; close all;

% Function to be minimised:
% y = 10sin(x) + (x+2)^2 + 10

% Set Up - Use arbitrary range [-10 5].
xValues = -10:0.1:5;
yValues = 10 * sin(xValues) + (xValues + 2).^ 2 + 10;

% Define parameters.
initialValue = -10;
iteratioinCounter = 1;

% During tuning the optimal parameters found were:
N = 150;
% Define sigma for normally distributed random number.
sigma = 1;

% Prepare plotting axes.
figure;
dimensions = [0.5 0.4 0.3 0.3];
plot(xValues, yValues);

% Load initial values
x_k = initialValue;
y_k = 10 * sin(x_k) + (x_k + 2).^ 2 + 10;

% Perform Simulated Annealing for N iterations

% Learning annotation.
txt = annotation('textbox', dimensions, 'String', 'learning ...', ...
        'FitBoxToText', 'on');

for i=1:N
    
    % Plot title for iteration.
    title(strcat('Question 3.c: Simulated Annealing iteration: ', ...
        num2str(iteratioinCounter)));

    hold on;

    % Plot current iteration.
    plot(x_k, y_k, 'r*', 'MarkerSize', 8);

    pause(0.1);

    % Define temperature for current iteration.
    T = 1 - (i/N);

    % Generate Proposal.
    x_p = x_k + normrnd(0, sigma, 1);

    % Compute Acceptance Probability.
    y_k = 10 * sin(x_k) + (x_k + 2).^ 2 + 10;
    y_p = 10 * sin(x_p) + (x_p + 2).^ 2 + 10;

    P = exp(-(y_p - y_k)/ T);

    % Compare acceptance to random number
    if P > rand
        x_k = x_p;
    end

    % Define next iteration.
    % No need to define x_k = x_k.

    y_k = 10 * sin(x_k) + (x_k + 2).^ 2 + 10;
    iteratioinCounter = iteratioinCounter + 1;

    hold off;

    % All plots included in axes to show model 'learning' over time.

end

% When N reached notify compltete.
delete(txt);

hold on;
message = 'Simulated Annealing complete !';
finalVal = strcat('Minimised value of x:', num2str(x_k));
str = {message, finalVal};

% Text box.
annotation('textbox', dimensions, 'String', str, 'FitBoxToText', 'on');
% Plot final value.
plot(x_k, y_k, 'g*', 'MarkerSize', 16, 'LineWidth', 2);

hold off;

%% Question 4a (Surface Plot)
clear; close all;

% Function to be plotted:
% z = 5(y - x^2)^2 + (3-x)^2

% Model function (Vectorised).
f = @(x, y) 5 * (y - x.^2).^2 + (3 - x).^2;

% Surface plot.
figure;
fsurf(f);

%% Question 4b (Gradient Descent)
clear; close all;

% Function to be minimised:
% z = 5(y - x^2)^2 + (3-x)^2

% Gradient (Jacobian Matrix):
% g = [-20x(y-x^2)-2(3-x); 10(y-x^2) ]

% Define parameters.
initial_x = 0;
initial_y = 5;
learningRate = 0.02;

% Threshold is an infintesimal value.
threshold = 1e-5;
iteratioinCounter = 1;

% Function definition
f = @(x, y) 5 * (y - x.^2).^2 + (3 - x).^2;

% Prepare plotting axes.
figure;
dimensions = [0.5 0.5 0.3 0.3];

% Load initial values.
x_k = initial_x;
y_k = initial_y;
z_k = 5 * (y_k - x_k.^2).^2 + (3 - x_k).^2;

% Calculate gradient.
g_k = [-20 * x_k * (y_k - x_k^2) - 2 * (3 - x_k); 10 * (y_k - x_k^2)]; 

% Perfrom Gradient Descent whilst gradient value is 
% above threshold.

% Learning annotation.
txt = annotation('textbox', dimensions, 'String', 'learning ...', ...
        'FitBoxToText', 'on');

while abs(g_k) > threshold
    % Plot function.
    fsurf(f);

    hold on;

    % Plot title for iteration.
    title(strcat('Question 4.b: Gradient Descent iteration: ', ...
        num2str(iteratioinCounter)));

    % Plot current value.
    scatter3(x_k, y_k, z_k, 'r*');

    pause(0.01);

    % Compute next iteration of descent:

    % Calculate gradient.
    g_k = [
            -20 * x_k * (y_k - x_k^2) - (2 * (3 - x_k)); ...
            10 * (y_k - x_k^2)
          ]; 

    % Perform descent function.
    x_k = x_k - (learningRate * g_k(1));
    y_k = y_k - (learningRate * g_k(2));

    % Load value for next iteration.
    z_k = 5 * (y_k - x_k.^2).^2 + (3 - x_k).^2;

    iteratioinCounter = iteratioinCounter + 1;

    % hold off;

    % All plots included in axes to show minimisation over time.
end

% When threshold reached notify complteted descent.
delete(txt);

hold on;
message = 'Gradient Descent complete !';
finalVal_x = strcat('Minimised value of x:', num2str(x_k));
finalVal_y = strcat('Minimised value of y:', num2str(y_k));
str = {message, finalVal_x, finalVal_y};

% Text box.
annotation('textbox', dimensions, 'String', str, 'FitBoxToText', 'on');
% Plot final value.
scatter3(x_k, y_k, z_k, 'g*');

hold off;

%% Question 4c (Newton's Method)
clear; close all;

% Function to be minimised:
% z = 5(y - x^2)^2 + (3-x)^2

% Gradient (Jacobian Matrix):
% g = [ -20x(y-x^2)-2(3-x); 10(y-x^2) ]

% Second derivative (Hessian Matrix):
% h = [ 60x^2-20y+2, -20x; -20x, 10 ]

% Define parameters.
initial_x = 0;
initial_y = 5;

% Threshold is an infintesimal value.
threshold = 1e-5;
iteratioinCounter = 1;

% Function definition
f = @(x, y) 5 * (y - x.^2).^2 + (3 - x).^2;

% Prepare plotting axes.
figure;
dimensions = [0.5 0.5 0.3 0.3];

% Load initial values.
x_k = initial_x;
y_k = initial_y;
z_k = 5 * (y_k - x_k.^2).^2 + (3 - x_k).^2;

% Calculate gradient.
g_k = [-20 * x_k * (y_k - x_k^2) - 2 * (3 - x_k); 10 * (y_k - x_k^2)];

% Calculate second derivative.
h_k = [60 * x_k.^2 - 20 * y_k + 2, -20 * x_k; -20 * x_k, 10];

% Perfrom Gradient Descent whilst gradient value is 
% above threshold.

% Learning annotation.
txt = annotation('textbox', dimensions, 'String', 'learning ...', ...
        'FitBoxToText', 'on');

while abs(g_k) > threshold
    % Plot function.
    fsurf(f);

    hold on;

    % Plot title for iteration.
    title(strcat('Question 4.c: Newtons Method iteration: ', ...
        num2str(iteratioinCounter)));

    % Plot current value.
    scatter3(x_k, y_k, z_k, 'r*');

    pause(0.5);

    % Compute next iteration:

    % Calculate gradient.
    g_k = [
            -20 * x_k * (y_k - x_k^2) - (2 * (3 - x_k)); ...
            10 * (y_k - x_k^2)
          ]; 

    % Calculate second derivative.
    h_k = [60 * x_k.^2 - 20 * y_k + 2, -20 * x_k; -20 * x_k, 10];

    % Perform descent function.
    % Note h^(-1) is used instead of inv(h) to avoid warning, they are
    % equivalent in operation.
    r = (h_k^(-1) * g_k);
    x_k = x_k - r(1);
    y_k = y_k - r(2);

    % Load value for next iteration.
    z_k = 5 * (y_k - x_k.^2).^2 + (3 - x_k).^2;

    iteratioinCounter = iteratioinCounter + 1;

    % hold off;

    % All plots included in axes to show minimisation over time.
end

% When threshold reached notify complteted descent.
delete(txt);

hold on;
message = 'Newtons Method complete !';
finalVal_x = strcat('Minimised value of x:', num2str(x_k));
finalVal_y = strcat('Minimised value of y:', num2str(y_k));
str = {message, finalVal_x, finalVal_y};

% Text box.
annotation('textbox', dimensions, 'String', str, 'FitBoxToText', 'on');
% Plot final value.
scatter3(x_k, y_k, z_k, 'g*');

hold off;

%% Question 4d (Simulated Annealing)
clear; close all;

% Function to be minimised:
% z = 5(y - x^2)^2 + (3-x)^2

% Note that for simulated annealing, multivariate normal
% distribution is used with mean vector u = [0 0].

% During tuning the optimal parameters found were:
% iterations: 500
% sigma_x: 3
% sigma_y: 0.5

% Define parameters.
initial_x = 0;
initial_y = 5;

% Simulated Annealing parameters.
iteratioinCounter = 1;
N = 500;
sigma_x = 3;
sigma_y = 0.5;
sigma_vec = [sigma_x sigma_y; sigma_y sigma_x];

% Function definition
f = @(x, y) 5 * (y - x.^2).^2 + (3 - x).^2;

% Prepare plotting axes.
figure;
dimensions = [0.5 0.5 0.3 0.3];

% Load initial values.
x_k = initial_x;
y_k = initial_y;
z_k = 5 * (y_k - x_k.^2).^2 + (3 - x_k).^2;

% Learning annotation.
txt = annotation('textbox', dimensions, 'String', 'learning ...', ...
        'FitBoxToText', 'on');

for i = 1:N
    % Plot function.
    fsurf(f);

    hold on;

    % Plot title for iteration.
    title(strcat('Question 4.d: Simulated Annealing iteration: ', ...
        num2str(iteratioinCounter)));

    % Plot current value.
    scatter3(x_k, y_k, z_k, 'r*');

    pause(0.01);

    % Compute next iteration:

    % Define temperature. 
    T = 1 - (i/N);

    % Compute random data from multivariate normal distribution.
    n = mvnrnd([0 0], sigma_vec);

    % Compute current value.
    z_k = 5 * (y_k - x_k.^2).^2 + (3 - x_k).^2;

    % Compute proposal
    x_p = x_k + n(1);
    y_p = y_k + n(2);

    z_p = 5 * (y_p - x_p.^2).^2 + (3 - x_p).^2;

    % Compute acceptance probablity.
    P = exp(-(z_p - z_k)/T);

    % Evaluate acceptance probablity
    if P > rand
        x_k = x_p;
        y_k = y_p;
    end

    % Load value for next iteration.
    z_k = 5 * (y_k - x_k.^2).^2 + (3 - x_k).^2;
    iteratioinCounter = iteratioinCounter + 1;

    % All plots included in axes to show minimisation over time.
end

% When threshold reached notify complteted descent.
delete(txt);

hold on;
message = 'Simulated Annealing complete !';
finalVal_x = strcat('Minimised value of x:', num2str(x_k));
finalVal_y = strcat('Minimised value of y:', num2str(y_k));
str = {message, finalVal_x, finalVal_y};

% Text box.
annotation('textbox', dimensions, 'String', str, 'FitBoxToText', 'on');
% Plot final value.
scatter3(x_k, y_k, z_k, 'g*');

hold off;

%% Question 5a (Plotting Data)
clear; close all;
load("Question5.mat");

% Scatter plot data.

% Class 1 - blue
% Class 2 - red

gscatter(data(:,1), data(:,2), L, 'br');
title('Question 5.a Original Dataset');
legend('Class 1', 'Class 2');

%% Question 5b (Fit Model, Display Parameters)
clear; close all;
load("Question5.mat");

% Using mnrfit, obtain model parameters.

% Prepare data.
L_categorical = categorical(L);

% Use mnrfit to perform regression.
W = mnrfit(data, L_categorical);

% Regression takes the form:
% lambda = 1 / (1 + exp(-(a*x_1 + b*x_2 + c)))

% Display data

figure;
dimensions = [0.3 0.3 0.3 0.3];

message = 'Logistic Regression Parameters';
firstparam = strcat('First parameter (a):', num2str(W(3)));
secondparam = strcat('Second parameter (b):', num2str(W(2)));
thirdparam = strcat('Third parameter (c):', num2str(W(1)));
str = {message, firstparam, secondparam, thirdparam};

annotation('textbox', dimensions, 'String', str, 'FitBoxToText', 'on', ...
    'FontSize', 18);

title('Question 5.b Matlab Model Parameters');

%% Question 5c (Show predictions from Logistic Regression)
clear; close all;
load("Question5.mat");

% For clarity, data is recalculated.
L_categorical = categorical(L);
W = mnrfit(data, L_categorical);

% Regression takes the form:
% lambda = 1 / (1 + exp(-(a*x_1 + b*x_2 + c)))

% Compute probability of class using model;
P = 1 ./ (1 + exp(-(data(:,1) .* W(3) + data(:,2) .* W(2) + W(1))));

% Assign class labels using probability;
for i = 1:length(P)

    if(P(i) > 0.5)
        L_model(i) = 0;
    else 
        L_model(i) = 1;
    end

end

% Plot data using labels from model.

% Class 1 - blue
% Class 2 - red

figure; 

% Subplot 1, original data.
subplot(2,1,1);
gscatter(data(:,1), data(:,2), L, 'br');
title('Question 5.c: Original Data');
legend('Class 1', 'Class 2');

subplot(2,1,2);
gscatter(data(:,1), data(:,2), L_model, 'br');
title('Question 5.c: Matlab Modelled Dataset Labels');
legend('Class 1', 'Class 2');

%% Question 5d (Confusion Matrix)

% Data is regenerated according to Q5c for clarity.
clear; close all;
load("Question5.mat");

L_categorical = categorical(L);
W = mnrfit(data, L_categorical);
P = 1 ./ (1 + exp(-(data(:,1) .* W(3) + data(:,2) .* W(2) + W(1))));

% Assign class labels using probability;
for i = 1:length(P)

    if(P(i) > 0.5)
        L_model(i) = 0;
    else 
        L_model(i) = 1;
    end

end

% Confusion Matrix
cm = confusionchart(L_model, L);

%% Question 6a (Neural Network - Training)
clear; close all;
load("Question5.mat");

% Using patternnet and train, model 2 layer 
% feedforward Neural Network.

% Define sizes for hidden layers.
% First layer -> Node for each point in the dataset.
% Second layer -> Node for each quadrant in the dataseet 
% (observed from plotting data in question 5).
net = patternnet([270 4]);

% Default training function, it is set explicitly 
% to acknowledge the testing of other training 
% funcions.
net.trainFcn = 'trainscg';

% Adjust network parameters.
% After tuning the following parameters provided the 
% best performance.
net.trainParam.lr = 0.01;
net.trainParam.epochs = 20;

% Train network.
trained_net = train(net, data', L');

%% Question 6b (Neural Network - Predictions)

% NB: Uses data from section 6a.

% Use model trained in part a for predicitons.
pred = trained_net(data');
nn_Predictions = pred';

% Assign class labels using probability;
for i = 1:length(nn_Predictions)

    if(nn_Predictions(i) > 0.5)
        L_model(i) = 1;
    else 
        L_model(i) = 0;
    end

end

figure;

% Subplot 1, original data.
subplot(2,1,1);
gscatter(data(:,1), data(:,2), L, 'br');
title('Original Data');


% Subplot 2, data from model.
subplot(2,1,2);
gscatter(data(:,1), data(:,2), L_model, 'br');
title('Data From Neural Network Model');

%% Question 6c (Neural Network - Confusion Matrix)

% NB: Uses data from section 6a.
% NB: Uses data from section 6b.

close all;

% Confusion Matrix
cm = confusionchart(L_model, L);

%% Question 7 (Logistic Regression with Kernels)
clear; close all;
load("Question5.mat");

% Based on characteristics of data, gaussian 
% kernel can use cluster centres, with 4 
% centres present we must extend the kernel
% to fit 4 dimensions. 

% After tuning the following parameters were
% found to be optimal.

% Centres.
mu_1 = [0 -1];
mu_2 = [5 4];
mu_3 = [7 7];
mu_4 = [11 -1];

% Sigma values.
sigma_1 = 0.1;
sigma_2 = 0.1;
sigma_3 = 0.1;
sigma_4 = 0.1;

% Kernel application
for i = 1:size(data, 1)
    k_1 = exp(-sigma_1 .* ((data(i,:) - mu_1) * (data(i,:) - mu_1)'));
    k_2 = exp(-sigma_2 .* ((data(i,:) - mu_2) * (data(i,:) - mu_2)'));
    k_3 = exp(-sigma_3 .* ((data(i,:) - mu_3) * (data(i,:) - mu_3)'));
    k_4 = exp(-sigma_4 .* ((data(i,:) - mu_4) * (data(i,:) - mu_4)'));
    k(i, :) = [k_1 k_2 k_3 k_4];
end

% Regression.
L_categorical = categorical(L);

W = mnrfit(k, L_categorical);
P = 1 ./ (1 + exp(-(k(:,1) .* W(5) + k(:,2) .* W(4)+ ...
    k(:,3) .* W(3) + k(:,4) .* W(2) + W(1))));
logical = (P > 0.5);
L_model = double(logical);

% Confusion Matrix
cm = confusionchart(L_model, L);

%% Question 8 (Logistic Regression using Gradient Descent)
clear; close all;
load("Question8.mat");

% Model parameters.
learningRate = 1e-4;
threshold = 1e-3;
iteratioinCounter = 1;

a_check = 1;
b_check = 1;
c_check = 1;
d_check = 1;

% Apply initial values.
a_k = 3;
b_k = 3;
c_k = 3;
d_k = 3;

% Plot original data set.
figure;
scatter3(x, y, z, 'b*');
dimensions = [0.6 0.5 0.3 0.3];

% Learning annotation.
txt = annotation('textbox', dimensions, 'String', 'learning ...', ...
        'FitBoxToText', 'on');

% Gradient descent loop - applied till threshold
% values reached.
while (a_check || b_check || c_check || d_check)
    scatter3(x, y, z, 'b*');
    % Plot title for iteration.
    title(strcat('Question 8: Gradient Descent iteration: ', ...
        num2str(iteratioinCounter)));

    hold on;

    % Compute points using regression.
    A = [x.^2 y.^2 x.*y ones(size(x,1),1)];
    W = [a_k; b_k; c_k; d_k];
    z_hat = A * W;

    % Plot data.
    scatter3(x, y, z_hat, 'r*');
    pause(0.1); 

    % Perfrom descent for next iteration.
    const = (z - (a_k .* x.^2 + b_k .* y.^2 + c_k .* x .* y + d_k));

    a_gradient = sum(- 2 .* (x.^2) .* const);
    a_k = a_k - learningRate .* a_gradient;
    a_check = (abs(a_gradient) > threshold);
    
    b_gradient = sum(- 2 .* (y.^2) .* const);
    b_k = b_k - learningRate .* b_gradient;
    b_check = (abs(b_gradient) > threshold);

    c_gradient = sum(- 2 .* (x.*y) .* const);
    c_k = c_k - learningRate .* c_gradient;
    c_check = (abs(c_gradient) > threshold);

    d_gradient = sum(- 2 .* ones(size(x,1), 1) .* const);
    d_k = d_k - learningRate .* d_gradient;
    d_check = (abs(d_gradient) > threshold);

    iteratioinCounter = iteratioinCounter + 1;
    hold off;
end

% When threshold reached notify complteted descent.
delete(txt);

hold on;
message = 'Gradient Descent complete !';
a_Val = strcat('Value of a:', num2str(a_k));
b_Val = strcat('Value of b:', num2str(b_k));
c_Val = strcat('Value of c:', num2str(c_k));
d_Val = strcat('Value of d:', num2str(d_k));

str = {message, a_Val, b_Val, c_Val, d_Val};

% Text box.
annotation('textbox', dimensions, 'String', str, 'FitBoxToText', 'on');
legend('Original Data', 'Model');

hold off;
