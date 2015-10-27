%%% Julian Anthony Brackins   
%%% CSC 514 - Computer Vision %%%
%%% Homework 7                %%%

function edge()

    clc
    close all

    %Image Files
    f1 = 'angry.png';
    f2 = 'bike.jpg';
    f3 = 'real.png';
    f4 = 'wolf.jpg'; 
    f5 = 'zooey.jpg';
    
    %KERNEL methods
    rob = 'ROBERTS';
    pre = 'PREWITT';
    sob = 'SOBEL';
    iso = 'ISOTROPIC';
    gau = 'GAUSSIAN';
  
    %Convert each image to grayscale doubles
    %See prepimage() function below
    [ img01, img01g ] = prepimage( f1 );
    [ img02, img02g ] = prepimage( f2 );
    [ img03, img03g ] = prepimage( f3 );
    [ img04, img04g ] = prepimage( f4 );
    [ img05, img05g ] = prepimage( f5 );

    %Generate Kernels
    %See get_kernel() function below
    [ k01x, k01y ] = get_kernel(rob); 
    [ k02x, k02y ] = get_kernel(pre); 
    [ k03x, k03y ] = get_kernel(sob); 
    [ k04x, k04y ] = get_kernel(iso); 
    [ k05x, k05y ] = get_kernel(gau, 0.8 ); 

    %Generate Images (x derivative, y derivative, edge map)
    %See get_edge() function below
    [ map01x, map01y, edge01 ] = get_edge( img01g, k01x, k01y );
    [ map02x, map02y, edge02 ] = get_edge( img02g, k02x, k02y );
    [ map03x, map03y, edge03 ] = get_edge( img03g, k03x, k03y );
    [ map04x, map04y, edge04 ] = get_edge( img04g, k04x, k04y );
    [ map05x, map05y, edge05 ] = get_edge( img05g, k05x, k05y );

    %Display Images (original image, x derivative, y derivative, edge map)
    %see genfigure() function below
    genfigure( img01, map01x, map01y, edge01, rob);
    genfigure( img02, map02x, map02y, edge02, pre);
    genfigure( img03, map03x, map03y, edge03, sob);
    genfigure( img04, map04x, map04y, edge04, iso);
    genfigure( img05, map05x, map05y, edge05, strcat(gau, ' | SIGMA = 0.8'));

end

function [ img, img_g ] = prepimage( file )
    %Read in image, scale it down a bit, return colored img and grayscale
    img = imread( file );
    img = imresize( img, 0.8 );
    img_g = rgb2gray( img );
    img_g = double( img_g );
end

function genfigure( i, ix, iy, iedge, descr )

    %Generate Images (original image, x derivative, y derivative, edge map)
    set(gca,'LooseInset',get(gca,'TightInset'))

    %Show images with appropriate descriptions
    figure, imshow(i,[]);    
    title(strcat('Original Image: ', descr));
    
    figure, imshow(ix,[]);    
    title(strcat('X Derivative: ', descr));

    figure, imshow(iy,[]);
    title(strcat('Y Derivative: ', descr));

    figure, imshow(iedge,[]);
    title(strcat('Edges: ', descr));

end

function [ G ] = convolution( scene, kernel )
    %%For Convolution, just Flip the filter in both directions
    %%Then apply Correlation as before...
    %%BTW rot90(x,2) is faster than flipud() + fliplr() which also works...
    kernel = rot90(kernel,2);
    
    %%Find the dimensions for the scene and kernel
    [kH, kW] = size(kernel);
    [sH, sW] = size(scene);

    %Get Half-Dimensions of Kernel to determine padding on each side
    kH2 = floor(kH / 2);
    kW2 = floor(kW / 2);

    %GENERATE HORIZONTAL PADDING
    pad = zeros(kH2,sW);
    %PAD TOP OF IMAGE
    scene = vertcat(pad,scene);
    %PAD BOTTOM OF IMAGE
    scene = vertcat(scene,pad);
    
    %GENERATE VERTICAL PADDING
    pad = zeros(sH + 2 * kH2,kW2);
    %PAD LEFT SIDE OF IMAGE
    scene = horzcat(pad,scene);
    %PAD RIGHT SIDE OF IMAGE
    scene = horzcat(scene,pad);


    %%set F, G, H matrices so that they match what's in the book.
    G = double(zeros(sH, sW));
    F = scene;
    H = kernel;

    %Calculate Correlation. Fixed so the summation appears closer to slides
    for i = 1:sH
        for j = 1:sW
        total = 0;
            for u = 1:kH
                for v = 1:kW
                    total = total + H(u,v) * F(i + u - 1,j + v - 1);
                end
            end
            G(i,j) = total / (sH * sW * kH * kW);
        end
    end
    
end

function [ Gx, Gy ] = get_kernel( kType, kSigma )
    %Ensure first arg is always caps
    kType = upper(kType);
    
    %Only Gaussian has kSigma
    if nargin < 2
       kSigma = 0;
    end


    
    if strcmp( kType, 'ROBERTS')
        %ROBERTS:
        Gx = [  0  1  ;
               -1  0 ];
        Gy = [  1  0  ;
                0 -1 ];
        
    else
        %HANDLE THE OTHERS
        if strcmp( kType, 'PREWITT')
            %PREWITT:
            p = 3;
        
        elseif strcmp( kType, 'SOBEL')
            %SOBEL:
            p = 4;
        
        elseif strcmp( kType, 'ISOTROPIC')
            %ISOTROPIC:
            p = 2 + sqrt(2);

        elseif strcmp( kType, 'GAUSSIAN')
            %GAUSSIAN:
            p =  2 * pi * ( kSigma * kSigma ) ;
       
        end
        
        if strcmp( kType, 'GAUSSIAN')
        %Gradient Kernel Family used for Gaussian:
        GradKernX = [   -1    0    1   ;
                        -1    0    1   ;
                        -1    0    1  ];

        GradKernY = [   -1   -1   -1   ;
                         0    0    0   ;
                         1    1    1  ];                     
        else
        %Gradient Kernel Family used for Prewitt, Sobel, Iso:
        GradKernX = [   -1    0    1   ;
                        2-p   0   p-2  ;
                        -1    0    1  ];

        GradKernY = [   -1   2-p  -1   ;
                         0    0    0   ;
                         1   p-2   1  ];            
        end

        %Apply finite difference kernels
        Gx = (1/p) * GradKernX;
        Gy = (1/p) * GradKernY;
        
    end
end

function [ derX, derY, edge ] = get_edge( image, kernX, kernY )
    %Generate X Derivative Map through convolution
    derX = convolution( image, kernX );
    %Generate Y Derivative Map through convolution
    derY = convolution( image, kernY );
    %Generate Edge Map through gradient magnitude
    edge = sqrt(derX.^2 + derY.^2);
end