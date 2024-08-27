%% Q7 proofs 

%% Question 7 (Improved Logistic Regression)
clear; close all;
load("Question5.mat");

% Blue
% Gaussian kernel properties
centre_1 = [-1 -1];
centre_2 = [8 8];

B_1 = 0.6;
B_2 = 0.4;


centre_Vector_1 = repmat(centre_1, 270, 1);
centre_Vector_2 = repmat(centre_2, 270, 1);

k = exp(-B_1 * (data - centre_Vector_1).^2) + ...
exp(-B_2 * (data - centre_Vector_2).^2);

gscatter(data(:,1), k(:,1), L);

%%
c1 = [-1 -1];
c2 = [8 7];

B1 = 0.6;
B2 = 0.4;

cv1 = repmat(c1, 270, 1);
cv2 = repmat(c2, 270, 1);

cvs = repmat([9 7], 270, 1);
Bs = 0.3;
s = 1 ./ (1 + exp(-Bs .* (data + cvs)));


g1 = exp(-B1 * (data - cv1).^2);
g2 = exp(-B2 * (data - cv2).^2) .* s;



k = g1 + g2;


gscatter(data(:,1), k(:,1), L);

%%
% Red

centre_1 = [11 -1];
centre_2 = [4 5];
centre_1b = [-1 -1];
centre_2b = [8 8];

B_1 = 0.3;
B_2 = 0.5;
B_1b = 0.6;
B_2b = 0.2;

cv1 = repmat(centre_1, 270, 1);
cv2 = repmat(centre_2, 270, 1);
cv1b = repmat(centre_1b, 270, 1);
cv2b = repmat(centre_2b, 270, 1);


k = exp(-B_1 * (data - cv1).^2) + ...
exp(-B_2 * (data - cv2).^2) + exp(-B_1b * (data - cv1b).^2) + ...
exp(-B_2b * (data - cv2b).^2);

gscatter(data(:,2), k(:,2), L);


%% 

centre_1 = [-1 -1];
centre_2 = [8 8];

B_1 = 0.6;
B_2 = 0.4;


centre_Vector_1 = repmat(centre_1, 270, 1);
centre_Vector_2 = repmat(centre_2, 270, 1);


k = exp(-B_1 * (data - centre_Vector_1).^2) + ...
exp(-B_2 * (data - centre_Vector_2).^2);

L_categorical = categorical(L);
W = mnrfit(k, L_categorical);
P = 1 ./ (1 + exp(-(k(:,1) .* W(3) + k(:,2) .* W(2) + W(1))));
logical = (P < 0.5);
L_model = double(logical);

% Confusion Matrix
figure;
cm = confusionchart(L_model, L);
title('Q7');


%% Bisector line - Graph
clear; close all;
load("Question5.mat");

figure;
gscatter(data(:,1), data(:,2), L, 'br');

hold on; 

x = -5: 0.1: 15;
yphi = (-8./7) .* x + 12;
ytheta = (-14./9) .* x + (34./9);

plot(x, yphi);
plot(x, ytheta);

%% Theta test 
clear; close all;
load("Question5.mat");

% Theta kernel
k = 1 ./ (1 + exp( 14.*data(:,1) + 9.*data(:,2) -34 ));

% gscatter(data(:,1), k(:,1), L);
gscatter(data, k, L);


%% Phi tuning
clear; close all;
load("Question5.mat");

B = 0.3;
c = [6 6];
cv = repmat(c, 270, 1);


k = (1 - (1 ./ (1 + exp( 8.*data(:,1) + 7.*data(:,2) - 84 ))));
s = 1 ./ (1 + exp(-B .* (data + cv)));
g = exp(-B .* (data - cv).^2);

% gscatter(data(:,1), g(:,1), L);
% gscatter(data(:,2), g(:,2),  L);
gscatter(data, k, L);

%% phi test 
clear; close all;
load("Question5.mat");

B = 0.3;
c = [6 6];
cv = repmat(c, 270, 1);

lambda = (1 - (1 ./ (1 + exp( 8.*data(:,1) + 7.*data(:,2) - 84 ))));
g = exp(-B .* (data - cv).^2);

k = lambda .* g;
gscatter(data, k, L);


%% Flipped 

% Theta kernel
k = 1 - (1 ./ (1 + exp( 14.*data(:,1) + 9.*data(:,2) -34 )));
gscatter(data(:,1), k(:,1), L);

%% Sigmoid kernel
clear; close all;
load("Question5.mat");

% Centres on blue 
B1 = 0.5;
B1g = 0.8;
c1 = [0 -2];
c1v = repmat(c1, 270, 1);

B2 = 0.1;
B2g = 0.2;
c2 = [8 7];
c2v = repmat(c2, 270, 1);


k = (1 ./ (1 + exp(-B1 .* (data + c1v)))) .* exp(B1g * (data - c1v).^2) ...
+ (1 ./ (1 + exp(-B2 .* (data + c2v)))) .* exp(B2g * (data - c2v).^2);

gscatter(data(:,1), k(:,1), L);



%% Applied kernel proof 
clear; close all;
load("Question5.mat");

% Theta kernel
theta = 1 ./ (1 + exp( 14.*data(:,1) + 9.*data(:,2) -34 ));


% Phi kernel 
B = 0.3;
c = [6 6];
cv = repmat(c, 270, 1);

% Separation
lambda = (1 - (1 ./ (1 + exp( 8.*data(:,1) + 7.*data(:,2) - 84 ))));
% Activation
g = exp(-B .* (data - cv).^2);

k = theta + lambda.*g;

gscatter(data, k, L);

%% Solution 1

clear; close all;
load("Question5.mat");

% Theta kernel
theta = 1 ./ (1 + exp( 14.*data(:,1) + 9.*data(:,2) -34 ));

% Phi kernel 
B = 0.3;
c = [6 6];
cv = repmat(c, 270, 1);

% Separation
lambda = (1 - (1 ./ (1 + exp( 8.*data(:,1) + 7.*data(:,2) - 84 ))));
% Activation
g = exp(-B .* (data - cv).^2);

k = theta + lambda.*g;

L_categorical = categorical(L);

W = mnrfit(k, L_categorical);
P = 1 ./ (1 + exp(-(k(:,1) .* W(3) + k(:,2) .* W(2) + W(1))));
logical = (P < 0.5);
L_model = double(logical);

% Confusion Matrix
cm = confusionchart(L_model, L);

%% Gaussian kernel 
load("Question5.mat");


% k = exp(-B1 .* (data - cv1).^2);
% theta
c1 = [-1 -1];
cv1 = repmat(c1, 270, 1);
B1 = 0.5;
k1 = exp( (-1 .* (data - cv1).^2) ./ (2.*B1).^2);
gscatter(data(:,1), k1(:,1), L);

 %% 
load("Question5.mat");

% phi 
c2 = [8 8];
cv2 = repmat(c2, 270, 1);
B2 = 0.6;

k2 = exp( (-1 .* (data - cv2).^2) ./ (2.*B2).^2);
gscatter(data(:,1), k2(:,1), L);

%% 
delta 
c2 = [8 7];
cv2 = repmat(c2, 270, 1);
B2 = 0.6;

k2 = exp( (-1 .* (data - cv2).^2) ./ (2.*B2).^2);
gscatter(data(:,1), k2(:,1), L);


%%
load("Question5.mat");

vec14 = repmat(14, 270, 1);
vec9 = repmat(9, 270, 1);

k = dot(vec14, data(:,1), 2) + dot(vec9, data(:,2), 2) -34;

gscatter(data(:,2), k, L);

%% Polynomial kernel 
load("Question5.mat");

p = 2;
c = 1;
k = (data(:,1) .* data(:,2)' + c).^p;

gscatter(data, k, L);

%% sigmoid kernel 
load("Question5.mat");

% Centres on blue 
B1 = 0.5;
c1 = [11 0];
c1v = repmat(c1, 270, 1);


% B2 = 0.1;
% B2g = 0.2;
% c2 = [8 7];
% c2v = repmat(c2, 270, 1);


k = (1 ./ (1 + exp(-B1 .* (data + c1v))));

gscatter(data(:,1), k(:,1), L);
% gscatter(data(:,2), k(:,2), L);

%% RBF Kernel 
load("Question5.mat");

% Centres on blue 
sigma1 = 0.55;
sigma2 = 0.5;
sigma3 = 0.05;


c1 = [8 8];
c2 = [0 -2];
c3 = [6 6];

c1v = repmat(c1, 270, 1);
c2v = repmat(c2, 270, 1);
c3v = repmat(c3, 270, 1);


k = exp( (-1 .* (data - c1v).^2) ./ (2.*sigma1).^2) + ...
exp( (-1 .* (data - c2v).^2) ./ (2.*sigma2).^2) + ...
exp( (-1 .* (data - c3v).^2) ./ (2.*sigma3).^2);

gscatter(data(:,1), k(:,1), L);

L_categorical = categorical(L);

W = mnrfit(k, L_categorical);
P = 1 ./ (1 + exp(-(k(:,1) .* W(3) + k(:,2) .* W(2) + W(1))));
logical = (P < 0.5);
L_model = double(logical);

% Confusion Matrix
cm = confusionchart(L_model, L);

%% 4D Gaussian
clear; close all;
load('Question5.mat');

sigma1 = 0.1;
sigma2 = 0.1;
sigma3 = 0.1;
sigma4 = 0.1;

c1 = [0 -1];
c2 = [5 4];
c3 = [7 7];
c4 = [11 -1];

for i = 1:size(data, 1)
    k_1 = exp(-sigma1 .* ((data(i,:) - c1) * (data(i,:) - c1)'));
    k_2 = exp(-sigma2 .* ((data(i,:) - c2) * (data(i,:) - c2)'));
    k_3 = exp(-sigma3 .* ((data(i,:) - c3) * (data(i,:) - c3)'));
    k_4 = exp(-sigma4 .* ((data(i,:) - c4) * (data(i,:) - c4)'));
    k(i, :) = [k_1 k_2 k_3 k_4];
end

% gscatter(data(:,1), k(:,1), L);

L_categorical = categorical(L);

W = mnrfit(k, L_categorical);
P = 1 ./ (1 + exp(-(k(:,1) .* W(5) + k(:,2) .* W(4)+ ...
    k(:,3) .* W(3) + k(:,4) .* W(2) + W(1))));
logical = (P > 0.5);
L_model = double(logical);

% Confusion Matrix
cm = confusionchart(L_model, L);