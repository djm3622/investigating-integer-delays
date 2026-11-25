function make_targets(kind, output_root, grid)
% MAKE_TARGETS  Write the comb or triangle targets of the letter.
%   make_targets('comb') writes data/vary_all_pluck and make_targets('triangle') writes
%   data/vary_all_triangle, next to this folder, where the Python loaders look for them.
%   Each target is a WAV (22025 Hz) and a MAT with its labels; the triangle set also gets
%   manifest.json. output_root overrides the data folder; grid (a struct with any of the
%   fields L, a, K, ratio, T) overrides the default parameter grid.
%   Peaks above 0.95 are scaled down and the scale is stored; the loaders reject such targets.
    if nargin < 2 || isempty(output_root)
        output_root = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data');
    end
    defaults = struct('L', 100:5:200, 'a', 0.1:0.1:0.9, 'K', linspace(0.7, 0.999, 18), ...
                      'ratio', [0.1, 0.3, 0.5], 'T', 40000);
    if nargin < 3
        grid = struct();
    end
    for field = fieldnames(defaults)'
        if ~isfield(grid, field{1})
            grid.(field{1}) = defaults.(field{1});
        end
    end

    Fs = 22025;
    T = grid.T;
    impulse_gain = 1.0;
    headroom = 0.95;
    switch kind
        case 'comb'
            output_dir = fullfile(output_root, 'vary_all_pluck');
        case 'triangle'
            output_dir = fullfile(output_root, 'vary_all_triangle');
        otherwise
            error('kind must be ''comb'' or ''triangle''.');
    end
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end
    if strcmp(kind, 'triangle')
        write_manifest(output_dir, Fs, T);
    end

    for delay_gain = grid.K
        for a = grid.a
            for L = grid.L
                for pluck_position_ratio = grid.ratio
                    p = round(pluck_position_ratio * L);
                    if strcmp(kind, 'comb')
                        [y, excitation] = comb_pluck(T, L, delay_gain, a, p, max(grid.L));
                    else
                        [y, excitation] = triangle_pluck(T, L, delay_gain, a, p);
                    end
                    peak_raw = max(abs(y));
                    scale = min(1.0, headroom / max(peak_raw, eps));
                    y = y * scale;
                    noise = excitation * scale;  % the input that produced the stored WAV

                    if strcmp(kind, 'comb')
                        pluck_delay = p;
                        stem = fullfile(output_dir, sprintf( ...
                            'vary_all_pluck_delay_gain_%.3f_a_%.1f_L_%03d_pos_%.1f_d_%03d', ...
                            delay_gain, a, L, pluck_position_ratio, p));
                        audiowrite([stem, '.wav'], y, Fs);
                        save([stem, '.mat'], 'noise', 'scale', 'peak_raw', ...
                             'pluck_delay', 'pluck_position_ratio', 'impulse_gain');
                    else
                        A = p;
                        stem = fullfile(output_dir, sprintf( ...
                            'vary_all_triangle_delay_gain_%.3f_a_%.1f_L_%03d_pos_%.1f_A_%03d', ...
                            delay_gain, a, L, pluck_position_ratio, p));
                        audiowrite([stem, '.wav'], y, Fs, 'BitsPerSample', 24);
                        save([stem, '.mat'], 'noise', 'scale', 'peak_raw', 'A', ...
                             'pluck_position_ratio', 'impulse_gain', 'L', 'a', 'delay_gain', 'Fs', 'T');
                    end
                end
            end
        end
    end
end

function write_manifest(output_dir, Fs, T)
% The Python loader checks this contract and removes exactly one leading sample.
    file = fopen(fullfile(output_dir, 'manifest.json'), 'w');
    if file < 0
        error('Could not write the triangle manifest.');
    end
    fprintf(file, ['{"sample_rate": %d, "causal_samples": %d, "leading_zero": 1, ' ...
                   '"sign": "negative", "audio_format": "24-bit PCM WAV"}\n'], Fs, T);
    fclose(file);
end
