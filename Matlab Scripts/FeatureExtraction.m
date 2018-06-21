%mfcc = zeros(1, 14);
%dlmwrite('Training Data.csv',zeros(1, 14),'delimiter',',');
files = dir('*.wav'); 
%Coeffs = zeros(1, 14);
[x, y] = size(files);
for i = 3 : x
    i
    [Y, fs] = audioread(files(i).name);
    %mfcc
    Coeffs = [mfcc(Y, fs) 0];
    %Coeffs = [Coeffs 1];
    if(~isnan(Coeffs(1:13)))
        dlmwrite('Testing Data.csv',Coeffs,'delimiter',',', '-append');
    end
end