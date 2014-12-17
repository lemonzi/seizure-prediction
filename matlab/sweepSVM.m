%% INIT

% We're gonna need a lot of memory...
clear all;
close all;
figure;

% Important! Window size, # bands, data, etc
settings = loadjson('./settings.json');
if isfield(settings, 'threshold') && ischar(settings.threshold)
    settings.threshold = eval(settings.threshold);
end

bands = [4, 8, 16, 32, 64];

for b = 1:length(bands)

    settings.bands.count = bands(b);

    acc = zeros(size(thresholds));

    [accuracy, results] = trainSVM(settings);
    for t = 1:length(thresholds)
        fprintf('##########\nThreshold: %f\n##########\n', thresholds(t));
        acc(t) = mean(accuracy{t});
    end

    plot(thresholds(1:t), acc(1:t), '.-', 'MarkerSize', 8, 'Color', niceColors(b));
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
