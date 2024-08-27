%% Question 8 (Logistic Regression using Gradient Descent)
clear; close all;
load("Question8.mat");

% Model parameters.
learningRate = 0.03;
threshold = 1e-2;
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
dimensions = [0.6 0.6 0.3 0.3];

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

    a_gradient = sum(- 2 .* (x.^2) .* const)/length(x);
    a_k = a_k - learningRate .* a_gradient;
    a_check = (abs(a_gradient) > threshold);
    
    b_gradient = sum(- 2 .* (y.^2) .* const)/length(x);
    b_k = b_k - learningRate .* b_gradient;
    b_check = (abs(b_gradient) > threshold);

    c_gradient = sum(- 2 .* (x.*y) .* const)/length(x);
    c_k = c_k - learningRate .* c_gradient;
    c_check = (abs(c_gradient) > threshold);

    d_gradient = sum(- 2 .* ones(size(x,1), 1) .* const)/length(x);
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

hold off;