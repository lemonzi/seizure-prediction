clear all;
close all;

filename = '../../Workshop - Shared/data/Dog_1_interictal_segment_0001.mat';
settings = loadjson('./settings.json');
bands = processSample(filename, settings);

[nch, nbands, nwin] = size(bands);

figure;
mkdir('out');
for i = 1:nwin
    imagesc(bands(:,:,i));
    axis off;
    axis square;
    export_fig(sprintf('out/img_%03d',i), '-transparent');
    clf;
end
!ffmpeg -framerate 3 -i out/img_%03d.png -c:v libx264 -r 30  out.mp4
!rm out/*.png
