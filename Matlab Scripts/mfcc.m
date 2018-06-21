function [Coeffs] = mfcc(Y, fs)

%initialisation
N = fs * 0.025; %Number of points in a 25ms frame
M = fs * 0.010; %10ms shift to the next frame
lowFreq = 300;  % Lower frequency of Mel filterbank in Hz
highFreq = 8000;  % Upper frequency of Mel filterbank in Hz
numFilterBanks = 26; %Standard value of number of filter banks

%Pad Y to get an even number of frames
numFrames = ceil((length(Y) - N)/M);
pad = zeros(1, N + numFrames*M - length(Y));
Y = [Y' pad];

%Divide into frames;
frames = zeros(numFrames, N);
k = 1;
for i = 1: numFrames
    frames(i, :) = Y(k : k + N-1).*hamming(N)';
    k = k + M;
end 
%multiply by hamming window
%frames = frames .* hamming(N)';

%Take DFT of frame
ffts = fft(frames');

%Find periodogram-based power spectral estimate for the speech frame
powerFrame = 1/N * abs(ffts).^2;

%Compute mel filterbank
lowMel = 2595 * log(1 + lowFreq/700);
highMel = 2595 * log(1 + highFreq/700);
m = linspace(lowMel, highMel, numFilterBanks);
h = 700 * (exp(m/2595) - 1);
f = floor((N+1)*h/fs);
filterBank = zeros(numFilterBanks, N);
%F = linspace( f_min, f_max, length(f));
%Generate Filter banks
for i = 1:numFilterBanks-2
   
    for k = 1 : i
        filterBank(i, k) = 0;
    end 
    for k = i : i+1
       filterBank(i, k) = (k - f(i))/(f(i+1) - f(i));
    end 
    for k = i+1 : i+2
        filterBank(i, k) = (f(i+2) - k)/(f(i+2) - f(i+1));
    end
    for k = i+2 : length(f)
        filterBank(i, k) = 0;
    end 
  %  k = F >= f(i) & F <= f(i+1);
  %  filterBank(i, k) = (k - f(i-1))/(f(i) - f(i-1));
 %    k = F >= f(i+1) & F <= f(i+2);
 %   filterBank(i, k) = (k - f(i-1))/(f(i) - f(i-1));
end
%size(filterBank);
%size(powerFrame);
%Calculate filterbank energies
%bankEnergy = zeros(numFilterBanks, 1);
bankEnergy = sum(filterBank*powerFrame);

%Find logarithms
bankEnergy = log(bankEnergy);

%Find DCT
Coeffs = abs(dct(bankEnergy));
Coeffs = Coeffs(1:13);

%end 