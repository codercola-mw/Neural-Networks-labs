function [ P, T, F, A ] = Ada( nK, x, y )
%ADABOOST implements the Adaboost algorithm.
%   INPUTS:
%    - nK: Number of classifiers.
%    - x: Input samples.
%    - y: Output samples.
%   OUTPUTS:
%    - P: Polarity of each classifier.
%    - T: Threshold of each classifier.
%    - F: Feature of each classifier.
%    - A: Performance of each classifier (alpha).

    [ nF, nT ] = size(x);

    P = zeros(nK,1); %Polarity
    T = zeros(nK,1); %Threshold
    F = zeros(nK,1); %Feature used
    E = ones(nK,1)*inf; %Error per classifier
    A = zeros(nK,1); %Performance (alpha)

    % Importance of each example
    d = ones(1, nT)/nT;

    % Iterate each classifier looking for the one with lower error
    for k = 1:nK
        h_best = 0;

        % Iterate all features
        for f = 1:nF
            % Iterate all thresholds (data points)
            for t = 1:nT
                % Calculate error of each classifier
                p = 1;
                t_value = x(f,t);

                h_x = weakclassifier(tm,pm,xTrain);    
                e = sum(d .* (y ~= h_x));
                if e > 0.5
                    e = 1-e;
                    p = -1;
                    h_x = -h_x;
                end

                % Look for the weak classifier that minimizes the error
                if e < E(k)
                    E(k) = e;
                    P(k) = p;
                    T(k) = t_value;
                    F(k) = f;
                    h_best = h_x;
                end
            end
        end    

        % Update weights and renormalize
        A(k) = 0.5 * real(log((1-E(k))/E(k)));
        d = d .* exp(-A(k) * (y.*h_best) );
        d = d / sum(d);

    end

end

