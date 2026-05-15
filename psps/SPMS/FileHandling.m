clear
clc
filePattern = fullfile('./','*.psp8');
files = dir(filePattern);

for i = 1:length(files)
    filename = files(i).name;
    newname = filename(1:2);
    movefile(fullfile('./',filename), fullfile('./',strcat(newname,'.psp8')));
end