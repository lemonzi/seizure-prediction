% Generates parameter lattice for the SVM

% Where to save stuff
outFolder = './lattice';

% Default settings
settings = loadjson('settings.json');

% Values for each dimension
lengths = [1,2,5,10];
bands = [4,8,16,24,32];
kernels = {'polynomial', 'radial', 'linear'};

oldFolder = pwd();
mkdir(outFolder);
cd(outFolder);

for len = lengths
    settings.window.length = len;
    for band = bands
        settings.bands.count = band;
        for kernel_idx = 1:numel(kernels)
            settings.svm.kernel = kernels{kernel_idx};

            savejson('',settings,'temp');
            !shasum temp | awk '{printf("%s %s.json",$2,$1)}' | xargs 'mv'

        end
    end
end

cd(oldFolder);