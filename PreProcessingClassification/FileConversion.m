

USC = dir('..\USC-HAD\');

for i = 1:length(USC)
    subject = USC(i).name;

    if startsWith(subject, 'Subject')
        path = ['..\USC-HAD\' subject];
        files = dir(path);

        for j = 1:length(files)
            filename = files(j).name;
            if ~startsWith(filename, '.')
                file_path = [path '\' filename];
                M = load(file_path);

                save_path = ['RawData2\' subject];
                reWrite(M, filename, save_path)
            end

        end
    end
end


function reWrite(M, filename, new_path)

file_parts = split(filename, ".");
file_prefix = file_parts{1};
new_filename = [new_path '_' file_prefix, '.csv'];

% M = load(filename);

sensor_headings = {'acc_x, w/ unit g', 'acc_y, w/ unit g', ...
'acc_z, w/ unit g', 'gyro_x, w/ unit dps', ...
    'gyro_y, w/ unit dps', 'gyro_z, w/ unit dps'};

T = array2table(M.sensor_readings, ...
    'VariableNames',sensor_headings);

writetable(T, new_filename)

disp([filename ' complete'])

end

