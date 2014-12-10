%% INIT

% We're gonna need a lot of memory...
clear all;
close all;
figure;

% Important! Window size, # bands, data, etc
settings = loadjson('./settings.json');

bands = [4, 8, 16, 32, 64];
thresholds = [1e-3 0.1 0.25 0.5 0.75 0.9 1-1e-3];

for b = 1:length(bands)

    settings.bands.count = bands(b);

    acc = zeros(size(thresholds));

    for t = 1:length(thresholds)
        fprintf('##########\nThreshold: %f\n##########\n', thresholds(t));
        settings.threshold = thresholds(t);
        [accuracy, results] = trainSVM(settings);
        acc(t) = mean(accuracy);
    end

    plot(thresholds(1:t), acc(1:t), '.-', 'MarkerSize', 10, 'Color', niceColors(b));
    xlim([-0.05 1.05]);
    ylim([0.6 1.05]);
    xlabel('Threshold');
    ylabel('Accuracy');
    legend(arrayfun(@(x)sprintf('%d bands',x), bands(1:b), 'UniformOutput', false));
    hold on;
    drawnow;

end


% type1(fold) = mean(result == 1);
% type2(fold) = mean(result == -1);
