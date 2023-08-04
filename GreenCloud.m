
img = imread("D:\OneDrive\Desktop\matlab project\images_2\0_0.jpg")

[bw , mad] = createmask(img)
percentage = AreaCoverage(mad)
disp(percentage)


function percentage = AreaCoverage(filtered)
     
      total_pixel = numel(filtered)
      green_pixel = length(filtered(filtered~=0))

      percentage = (green_pixel/total_pixel)*100
end



function [BW,maskedRGBImage] = createmask(RGB)

% Convert RGB image to chosen color space
I = rgb2hsv(RGB);

% Define thresholds for channel 1 based on histogram settings
channel1Min = 0.223;
channel1Max = 0.411;

% Define thresholds for channel 2 based on histogram settings
channel2Min = 0.050;
channel2Max = 1.000;

% Define thresholds for channel 3 based on histogram settings
channel3Min = 0.121;
channel3Max = 1.000;

% Create mask based on chosen histogram thresholds
sliderBW = (I(:,:,1) >= channel1Min ) & (I(:,:,1) <= channel1Max) & ...
    (I(:,:,2) >= channel2Min ) & (I(:,:,2) <= channel2Max) & ...
    (I(:,:,3) >= channel3Min ) & (I(:,:,3) <= channel3Max);
BW = sliderBW;

% Initialize output masked image based on input image.
maskedRGBImage = RGB;

% Set background pixels where BW is false to zero.
maskedRGBImage(repmat(~BW,[1 1 3])) = 0;

end
