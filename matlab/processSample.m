function data_bands = processSample(filename, settings)
    % Process the file located at {{filename}}, using {{settings}}
    % Data dimensions: (channel, time/frequency/band, window)

    data = load(filename);

    % Get rid of the sub-structure with the recording code (Dog_1_...)
    name = fieldnames(data);
    data = data.(name{1});

    % Windowing indexes
    if (settings.window.length < settings.window.overlap)
        error('Overlap greater than window length');
    end
    fs = data.sampling_frequency;
    [nch, nsamp] = size(data.data);
    winlen = round(fs * settings.window.length);
    if mod(winlen,2)
        % Make window length even so we can take half FFT window
        winlen = winlen + 1;
    end
    overlap = round(fs * settings.window.overlap);
    overlap = winlen - overlap;
    nwindows = floor(nsamp / overlap);

    % Cut data into chunks and generate windoed matrix
    raw = zeros(nch, winlen, nwindows);
    for i = 1:nwindows
        raw(:,:,i) = data.data(:, overlap*(i-1)+1 : overlap*(i-1)+winlen);
    end

    % 1st order FIR filter to normalize frequency energies
    normalized = diff(raw, 1, 2);

    % FFT magnitude analysis, discarding phase and DC
    nbins = winlen/2 - 1;
    freq = abs(fft(normalized, [], 2));
    freq = freq(:,2:nbins+1,:);

    % Band analysis (integrate over frequency with square windows)
    nbands = settings.bands.count;
    bands = round(exp(linspace(0,log(nbins),nbands+1)));

    data_bands = zeros(nch, nbands, nwindows);
    for i = 1:nbands
        data_bands(:,i,:) = sum(freq(:,bands(i):bands(i+1), :), 2);
    end

end