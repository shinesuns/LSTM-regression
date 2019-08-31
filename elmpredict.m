function Y = elmpredict(P_test,IW,B,LW,TF,TYPE)
% P   - Input Matrix of testing Set  (R*Q)
% IW  - Input Weight Matrix (N*R)
% B   - Bias Matrix  (N*1)
% LW  - Layer Weight Matrix (N*S)
% TF  - Transfer Function:
%       'sig' for Sigmoidal function (default)
%       'sin' for Sine function
%       'hardlim' for Hardlim function
% TYPE - Regression (0,default) or Classification (1)
% Output
% Y   - Simulate Output Matrix (S*Q)

% Regression:
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',0)
% Y = elmpredict(P,IW,B,LW,TF,TYPE)

% Classification
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',1)
% Y = elmpredict(P,IW,B,LW,TF,TYPE)

if nargin < 6
    error('ELM:Arguments','Not enough input arguments.');
end


% Calculate the Layer Output Matrix H
Q = size(P_test,2);
BiasMatrix = repmat(B,1,Q);
tempH = IW * P_test + BiasMatrix;

switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'sin'
        H = sin(tempH);
    case 'hardlim'
        H = hardlim(tempH);
end

% Calculate the Simulate Output
Y = (H' * LW)';


if TYPE == 1
    temp_Y = zeros(size(Y));
    for i = 1:size(Y,2)
        [max_Y,index] = max(Y(:,i));
        temp_Y(index,i) = 1;
    end
    Y = vec2ind(temp_Y); 
end%出售各类算法优化深度极限学习机代码392503054