clear all;
close all;
figure;

folder = '~/Projects/UAB-Seizure Prediction/Dog_1/';
files = dirPattern([folder '*preictal*.mat']);
settings = loadjson('./settings.json');
mkdir('out');
mkdir('videos');
setenv('PATH', [getenv('PATH'), ':/usr/local/bin']);

for i = 1:numel(files)

    % filename = '../../Workshop - Shared/data/Dog_1_interictal_segment_0001.mat';
    filename = files{i};
    bands = processSample([folder filename], settings);
    [nch, nbands, nwin] = size(bands);

    mkdir(['out/', filename(1:end-4)]);
    for j = 1:nwin
        imagesc(bands(:,:,j)');
        axis off;
        axis equal;
        export_fig(['out/', filename(1:end-4), sprintf('/img_%03d.png',j)], '-transparent');
        clf;
    end
    setenv('OUTPUT_VIDEO_FN', ['videos/', filename(1:end-4), '.mp4'])
    setenv('INPUT_FOLDER', ['out/', filename(1:end-4)])
    !ffmpeg -framerate 4 -i $INPUT_FOLDER/img_%03d.png -c:v libx264 -vf scale=320:240 -r 30 $OUTPUT_VIDEO_FN &

end
