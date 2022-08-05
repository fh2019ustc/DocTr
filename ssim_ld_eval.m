path_rec = "xxx";  % rectified image path
path_scan = './scan/';  % scan image path

tarea=598400;
ms1=0;
ld1=0;
ms2=0;
ld2=0;
sprintf(path_rec)
for i=1:65
    path_rec_1 = sprintf("%s%d%s", path_rec, i, '_1 copy_rec.png');  % rectified image path
    path_rec_2 = sprintf("%s%d%s", path_rec, i, '_2 copy_rec.png');  % rectified image path
    path_scan_new = sprintf("%s%d%s", path_scan, i, '.png');  % corresponding scan image path

    % imread and rgb2gray
    A1 = imread(path_rec_1);
    A2 = imread(path_rec_2);

%    if i == 64
%        A1 = rot90(A1,-2);
%        A2 = rot90(A2,-2);
%    end

    ref = imread(path_scan_new);
    A1 = rgb2gray(A1);
    A2 = rgb2gray(A2);
    ref = rgb2gray(ref);

    % resize
    b = sqrt(tarea/size(ref,1)/size(ref,2));
    ref = imresize(ref,b);
    A1 = imresize(A1,[size(ref,1),size(ref,2)]);
    A2 = imresize(A2,[size(ref,1),size(ref,2)]);

    % calculate
    [ms_1,ld_1] = evalUnwarp(A1,ref);
    [ms_2,ld_2] = evalUnwarp(A2,ref);
    ms1 = ms1 + ms_1;
    ms2 = ms2 + ms_2;
    ld1 = ld1 + ld_1;
    ld2 = ld2 + ld_2;
end

ms = (ms1 + ms2) / 130
ld = (ld1 + ld2) / 130
