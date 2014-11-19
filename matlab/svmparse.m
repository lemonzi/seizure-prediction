function out = svmparse(settings)
% Parses a configuration struct
% Very useful for configuring SVM through JSON files
% Accepted fields (case insensitive, hyphens ignored):
%   - type: {C-SVC, nu-SVC, one-class, epsilon-SVR, nu-SVR}
%   - kernel: {linear, polynomial, radial, sigmoid}
%   - degree: degree of the polynomial (for polynomial kernel)
%   - cache: MB allocated to the SVM cache
%   - shrinking: whether to use the shrinking heuristics or not

    types = {'csvc', 'nusvc', 'oneclass', 'epsilonsvr', 'nusvr'};
    kernels = {'linear', 'polynomial', 'radial', 'sigmoid'};

    out= '';

    if isfield(settings, 'type')
        type_string = strfilter(settings.type);
        [~,type] = ismember(type_string, types);
        out = sprintf('%s -s %d',  out, type - 1);
    end

    if isfield(settings, 'kernel')
        kernel_string = strfilter(settings.kernel);
        [~,kernel] = ismember(kernel_string, kernels);
        out = sprintf('%s -t %d',  out, kernel - 1);
    end
    
    if isfield(settings, 'cache')
        out = sprintf('%s -m %d',  out, settings.cache);
    end
    
    if isfield(settings, 'degree')
        out = sprintf('%s -d %d',  out, settings.degree);
    end
    
    if isfield(settings, 'shrinking')
        out = sprintf('%s -h %d',  out, settings.shrinking);
    end
    
end

function strout = strfilter(strin)

    strout = lower(strin);
    strout = strout(...
        strout ~= '-' & ...
        strout ~= '_' & ...
        strout ~= ' ' ...
    );

end