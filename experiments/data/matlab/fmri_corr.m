expr_list = [1, 2, 3, 4, 5,6,7 8,9,10 11, 12, 13, 14, 15, 16, 17, 18,19,20 21, 22, 23, 24, 25, 26, 27, 28];
expr_list = [2];

addpath('../../nongit/pwling')

for idx = 1:numel(expr_list)
    element = expr_list(idx);
    expr = num2str(element);
    load(strcat('../../data/fmri_sim/sim', expr, '.mat'))
    csvwrite(strcat('../../data/fmri_sim/ts_sim', expr, '.csv'), ts);
    csvwrite(strcat('../../data/fmri_sim/net_sim', expr, '.csv'), net);

    end
end