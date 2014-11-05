function data_bands = processSample(filename, settings)
    % Process the file located at {{filename}}, using {{settings}}
    % Data dimensions: (channel, time/frequency/band, window)

    data = load(filename);

    % Get rid of the sub-structure with the recording code (Dog_1_...)
    name = fieldnames(data);
    data = data.(name{1});

    % Windowing indexes
    fs = data.sampling_frequency;
    [nch, nsamp] = size(data.data);
    winlen = round(fs * settings.window.length);
    if mod(winlen,2)
        % Make sure we can take half window (FFT positive side)
        winlen = winlen + 1;
    end
    nwindows = floor(nsamp / winlen);
    overlap = round(fs * settings.window.overlap);

    % Cut data into chunks and generate windoed matrix
    raw = zeros(nch, winlen, nwindows);
    for i = 1:nwindows
        raw(:,:,i) = data.data(:, overlap*(i-1)+1 : overlap*(i-1)+winlen);
    end

    % 1st order FIR filter to normalize frequency energies
    normalized = diff(raw, 1, 2);

    % FFT magnitude analysis, discarding phase
    nbins = winlen/2;
    freq = abs(fft(normalized, [], 2));
    freq = freq(:,1:nbins,:);

    % Band analysis (integrate over frequency with square windows)
    nbands = settings.bands.count;
    bandw = floor(nbins / nbands);
    data_bands = zeros(nch, nbands, nwindows);
    for i = 1:nbands
        data_bands(:,i,:) = sum(freq(:,(i-1)*bandw+1:i*bandw+1, :), 2);
    end

end