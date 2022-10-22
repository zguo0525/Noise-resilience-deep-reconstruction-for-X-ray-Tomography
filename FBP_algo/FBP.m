% +
clear;
clc;

addpath('bin');
addpath('npy-matlab');

ParamSetting;
% -

load_projections = readNPY('../tot_proj_ic_1600.npy');
load_projections = load_projections(1:10000, :, :);
projections_expand = zeros(10000, 1600, 2, 256);
projections_expand(:, :, 1, :) = load_projections;
projections_expand(:, :, 2, :) = load_projections;

% +
for offset = [40, 50]

    for idx = 1:10000

        disp(idx);

        param.filter='hann';

        param.dang= 360/1601*offset;
        param.deg = 0:param.dang:(360 - param.dang); 

        param.deg = param.deg*param.dir;
        param.nProj = length(param.deg);

        projections = projections_expand(idx, :, :, :);
        projections = reshape(projections, 1600, 2, 256);
        proj = permute(projections, [3, 2, 1]);
        proj = proj(:, :, 1:offset:end);

        proj_filtered = filtering(proj, param);
        Reconimg = CTbackprojection(proj_filtered, param);

        Reconimg_tot(idx, :, :) = Reconimg;
    end

    name = strcat('../FBP_results/FDK-ic-1600-tot-offset-', num2str(offset), '.npy');
    writeNPY(Reconimg_tot, name);
    
end

% +
for offset = [40, 50]

    for photon = [32, 40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400, 500, 640, 800, 1000, 1280, 1600, 2000]

        name = strcat('../tot_proj_test_ic_1600_photon_', num2str(photon), '.npy');
        load_projections = readNPY(name);

        projections_expand = zeros(1000, 1600, 2, 256);
        projections_expand(:, :, 1, :) = load_projections;
        projections_expand(:, :, 2, :) = load_projections;

        Reconimg_tot = zeros(1000, 150, 150);

        for idx = 1:1000

            param.filter='hann';

            param.dang= 360/1601*offset;
            param.deg = 0:param.dang:(360 - param.dang); 

            param.deg = param.deg*param.dir;
            param.nProj = length(param.deg);

            projections = projections_expand(idx, :, :, :);
            projections = reshape(projections, 1600, 2, 256);
            proj = permute(projections, [3, 2, 1]);
            proj = proj(:, :, 1:offset:end);

            proj_filtered = filtering(proj, param);
            Reconimg = CTbackprojection(proj_filtered, param);

            Reconimg_tot(idx, :, :) = Reconimg;
        end

        name = strcat('../FBP_results/FDK-ic-1600-test-photon-', num2str(photon), '-offset-', num2str(offset), '.npy');
        writeNPY(Reconimg_tot, name);

    end
    
end
