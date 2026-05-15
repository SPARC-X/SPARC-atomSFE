clear
clc

files = dir('*.psp8');
for id = 1:length(files)
    [~, ~,ext] = fileparts(files(id).name);
    % rename = strcat(f(1:2),ext) ;
    % movefile(files(id).name, rename);

    filename = fullfile(files(id).name);
    fid = fopen(filename,'r') ;
    assert(fid~=-1,'Error: Cannot open pseudopotential file %s',filename);
    textscan(fid,'%s',1,'delimiter','\n') ;
    Z = fscanf(fid,'%f',1);
    % Z = fscanf(fid,'%f',1);
    rename = strcat(num2str(Z),ext) ; 
    movefile(files(id).name, rename);
end