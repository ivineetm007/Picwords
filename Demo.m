%%PART 1-----------------------------------------------------------
%IMAGE SILHOUETTE  GENERATION
%-------------------------------------------------------------------
% Mean shift Segmentation code is taken from 
% Alireza Asvadi
% Department of ECE, SPR Lab
% Babol (Noshirvani) University of Technology
% http://www.a-asvadi.ir
% 2013
%% clear command windows
clc
clear all
format short g
close all
%% input
str = input('ENTER THE IMAGE NAME: ','s');
Orig    = imread(str);
I=Orig;
rows=size(I,1);
cols=size(I,2);
numRows=size(I,1);
numCols=size(I,2);
wrapImage=ones(numRows,numCols)*255;

choice=input('Press \n1 if Input image is of full body \n0 If Input image is of Face of a person...\n ');
fprintf("An image will be displayed click on it to proceed");
%% parameters
% meanshift parameter
bw   = 0.07;                % Mean Shift Bandwidth
%% Segmentation 
[Ims, Nms,backPixels]   = Ms(I,bw);                    % Mean Shift (color)
%% show
figure()
% subplot(3,2,1); imshow(I);    title('Original'); 
Igr=rgb2gray(Ims);
%fprintf('Number of background pixels are %d\n',backPixels);
forePixels=rows*cols-backPixels;
ImsB=imbinarize(Igr);
per=0;
for i =1:rows
    for j =1:cols
        if(ImsB(i,j)==0)
            per=per+1;
        end        
    end
end
per=per*100/forePixels;
if(per>=80 && choice==0)
    Igr=adapthisteq(Igr);
    Igr2=Igr;
    for i =1:size(I,1)
        for j =1:size(I,2)
            if(Igr2(i,j)>=100)
                Igr2(i,j)=255;
            else
                Igr2(i,j)=(Igr2(i,j)*255)/100;
            end        
        end
    end    
else
    Igr2=Igr;
end
ImsB=imbinarize(Igr2);
% subplot(3,2,2); imshow(Igr);  title(['Segmentation , background separation and grayscale conversion |','Number of segments detected : ',num2str(Nms)]);
% subplot(3,2,3); imshow(ImsB);  title(['Binary image',' : ','Otsu Method Matlab Inbuiltz |',' Percentage of black pixels  in foreground ', num2str(per)]);
% Ibn=zeros(size(I,1),size(I,2));
for i =1:size(I,1)
    for j =1:size(I,2)
        if(ImsB(i,j)==1)
            Ibn(i,j)=255;
        end        
    end
end
Iblur = imgaussfilt(Ibn,.5);
% subplot(3,2,4); imshow(Iblur);    title('SILHOUETTE after blurring ');
imwrite(Igr,'grayscale.jpg')
imwrite(Ibn,'sillhoute.jpg')

%%PART 2----------------------------------------

%%----------------------------------------------

Igr_o=imread('grayscale.jpg');
Ibn=imread('sillhoute.jpg');
Ibn = im2double(Ibn);
Igr=Igr_o;
[L,N] = superpixels(I,100,'Compactness',12);
BW = boundarymask(L);
%%subplot(4,2,5);
% subplot(3,2,1);
% imshow(Igr);
% subplot(3,2,2);
% imshow(Ibn);
% subplot(3,2,5);
imshow(imoverlay(I,BW,'cyan'),'InitialMagnification',67);title('Segemntation of original image');
w = waitforbuttonpress;
%PATCH RANKING AND REMOVING IRRELEVANT PATCHES-----------------------------------------------
%--------------------------------------------------------------------------------------------
idx = label2idx(L);
numRows = size(I,1);
numCols = size(I,2);
maxlen=sqrt((numRows^2)+(numCols^2));
totpix=numRows*numCols;
relevantPatches=0;
patches=zeros(N,9);
for labelVal = 1:N
    pix=idx{labelVal};
    totPixels=length(pix);
    patches(labelVal,1)=labelVal;
    silPixels=0;
    centroidP=0;
    for i=1:length(idx{labelVal})
        centroidP=centroidP+pix(i);
        x=mod(pix(i),numRows);
        if(x==0)
        x=numRows;
        end
        y=ceil(pix(i)/numRows);
        if(Ibn(x,y)==0)
            silPixels=silPixels+1;
        end
    end
    centroidP=centroidP/totPixels;
    if(silPixels<totPixels/1.5)
        patches(labelVal,2)=0;
        for i=1:length(idx{labelVal})
            x=mod(pix(i),numRows);
            if(x==0)
                x=numRows;
            end
            y=ceil(pix(i)/numRows);
            I(x,y,1)=255;
            I(x,y,2)=255;
            I(x,y,3)=255;
            Igr(x,y)=255;
        end
    else
        relevantPatches=relevantPatches+1;
        patches(labelVal,2)=1;
        patches(labelVal,3)=centroidP;
        patches(labelVal,4)=totPixels;
        x=mod(pix(i),numRows);
            if(x==0)
                x=numRows;
            end
        %MAJOR AND MINOR AXIS
        p=zeros(numRows,numCols);
        for i=1:length(idx{labelVal})
            x=mod(pix(i),numRows);
            if(x==0)
                x=numRows;
            end
            y=ceil(pix(i)/numRows);
            p(x,y)=255;
        end
        p=imbinarize(p);
        stats = regionprops('table',p,'MajorAxisLength','MinorAxisLength','Orientation');
%         s = regionprops(BW,'centroid');
%         centroids = cat(1, s.Centroid);
%         f = figure;
%         imshow(BW)
%         hold on
%         plot(centroids(:,1),centroids(:,2), 'b*')
%         hold off
%         w = waitforbuttonpress;
%         %imwrite(p,strcat('patch/',int2str(labelVal),'.jpg'));
        y=ceil(pix(i)/numRows);
        patches(labelVal,5)=sqrt(((x-numRows/2)^2) + ((y-numCols/2)^2));
        patches(labelVal,6)=stats.MajorAxisLength;
        patches(labelVal,7)=stats.MinorAxisLength;
        patches(labelVal,8)=(100*patches(labelVal,6)/maxlen)+(1000000*patches(labelVal,4)/totpix)-(400*patches(labelVal,6)/maxlen);
        patches(labelVal,9)=stats.Orientation;
    end
end
%READING WORDS FROM FILE AND SORTING BY LENGTH OF WORDS--------------------------------------------
%--------------------------------------------------------------------
fid = fopen('most_freq_words.txt','r');
wordsList=strings(relevantPatches,1);
for i=1:relevantPatches
    word=fgetl(fid);
    if(word~="")
        wordsList(i)=word;
    end
end
wordsList=sort(wordsList);
wordList2=cell(relevantPatches,1);
sortedLength=zeros(relevantPatches,1);
for i=1:relevantPatches
    wordList2{i}=convertStringsToChars(wordsList(i));
    sortedLength(i,1)=i;
end
stringLengthsOfA = cellfun(@(x)length(x),wordList2);
sortedLength=[sortedLength stringLengthsOfA];
sortedLength=sortrows(sortedLength,2,'descend');
for i=1:relevantPatches
    wordsList(i)=wordList2{sortedLength(i,1)};
end
%PATCH AND TEXT MAPPING AND TEXT WARPING-----------------------------
%--------------------------------------------------------------------
patches=sortrows(patches,8,'descend');
for labelVal=1:relevantPatches
    pix=idx{patches(labelVal,1)};
    pImage=ones(numRows,numCols)*255;
        for i=1:length(pix)
            x=mod(pix(i),numRows);
            if(x==0)
                x=numRows;
            end
            y=ceil(pix(i)/numRows);
            pImage(x,y)=0;
        end
    se1 = strel('disk',10);
    %patchI=imerode(patchI,se);
    pImage=imopen(pImage,se1);
   %imwrite(pImage,strcat('patch/',int2str(labelVal),'.jpg'));
%%MERGING------------------------------------------------------------------
%--------------------------------------------------------------------------
    patchI=pImage;
%%GETTING BOUNDARY POINTS OF REGION AND ITS CENTER-----------------------
%%-----------------------------------------------------------------------
    patchI=imcomplement(patchI);
    patchI=imbinarize(patchI);
    stats = regionprops('table',patchI,'Centroid','MajorAxisLength','MinorAxisLength','Orientation');
    xCenter=stats.Centroid(2);
    yCenter=stats.Centroid(1);
%%Getting image from text--------------------------------------------------
%text output is in variable a, black box output is in variable aY.
    string = wordsList(labelVal);
    angle = stats.Orientation;
    centre_x = xCenter;
    centre_y = yCenter;
    a = ones(numRows,numCols)*255;
    aY = ones(numRows,numCols)*255;
% num_rows = int16(sqrt((stats.MinorAxisLength)*(stats.MajorAxisLength))); % row size in pixel
% num_cols = int16(sqrt((stats.MinorAxisLength)*(stats.MajorAxisLength))); % col size in pixel
    num_rows=stats.MinorAxisLength;
    num_cols=stats.MinorAxisLength;
    fig = figure;
    len = strlength(string);
    t = text(0,0, string ,'FontName','Franklin Gothic Heavy','FontSize',20,'FontWeight','bold');
% qwertyuiopasdfghjklzxcvbnm
    avg_size = 15.53;
    x_ratio = 1.27;
    set(t,'Units','pixels');
    set(t,'VerticalAlignment','top');
    set(t,'Position',[0 37]);


% % set(t,'Rotation',-45);
    frame_width = (len)*avg_size;
    F = getframe(gca,[0 0 round(frame_width*1.5) 30]);
    close(fig)

    c = F.cdata(:,:,1);
    [i,j] = find(c==0);
    X=ones(38,round(2*frame_width)+20)*255;
    ind = sub2ind(size(X),i,j);
    X(ind) = uint8(0);
    X=255-X;
    [allRows, allColumns]=find(X);
    X=X(1:max(allRows),1:max(allColumns));
    X=255-X;
    X = imresize(X,[num_rows num_cols]);

    Y = zeros(size(X));
    % imagesc(X)
    % colormap bone
    % X = imrotate(X,45);
    A = X;
    T = @(I) imrotate(I,angle);
%     %// Apply transformation
    TA = T(A);
    mask = T(ones(size(A)))==0;
    TA(mask) = 1;
%%// Show image
% imshow(TA);
% % a = imread('index.jpg');
    Y = T(Y);
    Y(mask) = 1;


    b = TA;
    size_box = size(TA);
    startrow = round(centre_x-size_box(1,1)/2);
    startcol = round(centre_y-size_box(1,2)/2);
    
    if(startrow<=0)
        startrow=1;
    end

    if(startcol<=0)
        startcol=1;
    end
    lastrowb=round(startrow+size(b,1)-1);
    lastcolb=round(startcol+size(b,2)-1);
    lastrowY=round(startrow+size(Y,1)-1);
    lastcolY=round(startcol+size(Y,2)-1);
    if(lastrowb>=numRows)
        lastrowb=numRows;
    end
    
    if(lastcolb>=numCols)
        lastcolb=numCols;
    end
    if(lastrowY>=numRows)
        lastrowY=numRows;
    end
    
    if(lastcolY>=numCols)
        lastcolY=numCols;
    end
    a(startrow:lastrowb,startcol:lastcolb) = b(1:lastrowb-startrow+1,1:lastcolb-startcol+1);
    aY(startrow:lastrowY,startcol:lastcolY) = Y(1:lastrowY-startrow+1,1:lastcolY-startcol+1);
    textImage=a;
    textBlockImage=aY;
     imshow(textImage);
%     w = waitforbuttonpress;
%     imshow(textBlockImage);
    se = strel('disk',1);
    textImage=imerode(textImage,se);
%DISPLAYING TEXT ON IMAGE
    imshow(imoverlay(patchI,imcomplement(imbinarize(textImage)),'cyan'));title(['over image']);
%     w = waitforbuttonpress;

%     w = waitforbuttonpress;
%subplot(2,2,1);
% imshow(textImage);
% w = waitforbuttonpress;
% %subplot(2,2,2);
% imshow(textBlockImage);
% w = waitforbuttonpress;
%%-------------------------------------------------------------------------
    patchI=imcomplement(patchI);
    textBlockImage=imbinarize(textBlockImage);
    mask = boundarymask(patchI);
    maskBlock=boundarymask(textBlockImage);
    boundaryList=[];
    boundaryListBlock=[];
    for i=1:size(mask,1)
        for j=1:size(mask,2)
            if mask(i,j)==1
                boundaryList=[boundaryList;i j];
            end
        end
    end
    for i=1:size(maskBlock,1)
        for j=1:size(maskBlock,2)
            if maskBlock(i,j)==1
                boundaryListBlock=[boundaryListBlock;i j];
            end
        end
    end
%fprintf("number of points are %d",size(boundaryList,1))
    Xintersect=[];
    Yintersect=[];
%     XintersectBlock=[];
%     YintersectBlock=[];
    for i=1:size(boundaryList,1)
        if boundaryList(i,2)==1 || boundaryList(i,2)==numCols
            Xintersect=[Xintersect;boundaryList(i,1) boundaryList(i,2)];
        end
        if boundaryList(i,1)==1 || boundaryList(i,1)==numRows
            Yintersect=[Yintersect;boundaryList(i,1) boundaryList(i,2)];
        end    
    end

% imshow(imoverlay(textBlockImage,maskBlock,'red'),'InitialMagnification',67);title('Segemntation of original image');
% w = waitforbuttonpress;

%     for i=1:size(boundaryListBlock,1)
%         if boundaryListBlock(i,2)==1 || boundaryListBlock(i,2)==numCols
%             XintersectBlock=[XintersectBlock;boundaryListBlock(i,1) boundaryListBlock(i,2)];
%         end
%         if boundaryListBlock(i,1)==1 || boundaryListBlock(i,1)==numRows
%             YintersectBlock=[YintersectBlock;boundaryListBlock(i,1) boundaryListBlock(i,2)];
%         end    
%     end
    numXintersect=size(Xintersect,1);
    numYintersect=size(Yintersect,1);
%     numXintersectBlock=size(XintersectBlock,1);
%     numYintersectBlock=size(YintersectBlock,1);

    if(numXintersect>0 && numXintersect>0)
    elseif(numXintersect>1)
        Xintersect=sortrows(Xintersect,1);
        minPoint=Xintersect(1,:);
        maxPoint=Xintersect(numXintersect,:);
        y=minPoint(1,2);
        for x=minPoint(1,1):maxPoint(1,1)
            mask(x,y)=1;
            boundaryList=[boundaryList;x y];
        end
    elseif(numYintersect>1)
        Yintersect=sortrows(Yintersect,2);
        minPoint=Yintersect(1,:);
        maxPoint=Yintersect(numYintersect,:);
        x=minPoint(1,1);
        for y=minPoint(1,2):maxPoint(1,2)
        mask(x,y)=1;
        boundaryList=[boundaryList;x y];
        end
    end
    boundaryList=sortrows(boundaryList,1);

% if(numXintersectBlock>0 && numXintersectBlock>0)
% elseif(numXintersectBlock>1)
%     XintersectBlock=sortrows(XintersectBlock,1);
%     minPointBlock=XintersectBlock(1,:);
%     maxPointBlock=XintersectBlock(numXintersectBlock,:);
%     y=minPointBlock(1,2);
%     for x=minPointBlock(1,1):maxPointBlock(1,1)
%     maskBlock(x,y)=1;
%     boundaryListBlock=[boundaryListBlock;x y];
%     end
% elseif(numYintersectBlock>1)
%     YintersectBlock=sortrows(YintersectBlock,2);
%     minPointBlock=YintersectBlock(1,:);
%     maxPointBlock=YintersectBlock(numYintersectBlock,:);
%     x=minPointBlock(1,1);
%     for y=minPointBlock(1,2):maxPointBlock(1,2)
%     maskBlock(x,y)=1;
%     boundaryListBlock=[boundaryListBlock;x y];
%     end
% end
% boundaryListBlock=sortrows(boundaryListBlock,1);
% 
% %subplot(2,2,3);
% imshow(imoverlay(textBlockImage,maskBlock,'red'),'InitialMagnification',67);title('Segemntation of original image');
% w = waitforbuttonpress;
%GETTING POINT OF INTERSECYION OF 8 LINES WITH REGION----------------------
%--------------------------------------------------------------------------
    angleList=zeros(4,1);
    angleList2=[90;135;180;225];
    angleList(1,1)=90+stats.Orientation;
    angleList(2,1)=angleList(1,1)+45;
    angleList(3,1)=angleList(2,1)+45;
    angleList(4,1)=angleList(3,1)+45;
    pointList=zeros(9,4);
    numBoundaryPoints=size(boundaryList,1);
    numBoundaryPointsBlock=size(boundaryListBlock,1);

    for i=1:4
        angle=angleList(i,1)*pi/180;
        angle2=angleList2(i,1)*pi/180;
        %disp(angle);
        %disp(angleList(i,1));
        valMat=zeros(numBoundaryPoints,2);
        valMatBlock=zeros(numBoundaryPointsBlock,2);
        for p=1:numBoundaryPoints
            valMat(p,1)=p;
            if(abs(angle-(pi/2))>.1)
                valMat(p,2)=abs( (boundaryList(p,2)-yCenter)-(tan(angle)*(boundaryList(p,1)-xCenter)) );
            else
                valMat(p,2)=abs( boundaryList(p,1)-xCenter );
            end
        end
    
    
        for p=1:numBoundaryPointsBlock
            valMatBlock(p,1)=p;
%             if(abs(angle2-(pi/2))>.1)
                valMatBlock(p,2)=abs( (boundaryListBlock(p,2)-yCenter)-(tan(angle)*(boundaryListBlock(p,1)-xCenter)) );
%             else
%                 valMatBlock(p,2)=abs( boundaryListBlock(p,1)-xCenter );
%             end
        end
    
        valMat=sortrows(valMat,2);
        valMatBlock=sortrows(valMatBlock,2);
    %disp(valMat);
        first=boundaryList(valMat(1,1),:);
        firstBlock=boundaryListBlock(valMatBlock(1,1),:);
        k=0;
        kBlock=0;
        for p=2:numBoundaryPoints
            val= ((boundaryList(valMat(p,1),2)-yCenter)-tan(angle+pi/2)*(boundaryList(valMat(p,1),1)-xCenter) )*((first(1,2)-yCenter)-tan(angle+pi/2)*(first(1,1)-xCenter) );
            if(val<0 && sqrt(((first(1,1)-boundaryList(valMat(p,1),1))^2) + ((first(1,2)-boundaryList(valMat(p,1),2))^2))>3)
                k=p;
                break;
            end
        end
        for p=2:numBoundaryPointsBlock
                val= ((boundaryListBlock(valMatBlock(p,1),2)-yCenter)-tan(angle+pi/2)*(boundaryListBlock(valMatBlock(p,1),1)-xCenter) )*((firstBlock(1,2)-yCenter)-tan(angle+pi/2)*(firstBlock(1,1)-xCenter) );
            if(val<0 && sqrt(((firstBlock(1,1)-boundaryListBlock(valMatBlock(p,1),1))^2) + ((firstBlock(1,2)-boundaryListBlock(valMatBlock(p,1),2))^2))>3)
                kBlock=p;
                break;
            end
        end
    
    %fprintf("SECOND POINT IS AT P=%d",k);
        pointList((2*i)-1,1)=first(1,1);
        pointList((2*i)-1,2)=first(1,2);
        second=boundaryList(valMat(k,1),:);
        secondBlock=boundaryListBlock(valMatBlock(kBlock,1),:);
        pointList(2*i,1)=second(1,1);
        pointList(2*i,2)=second(1,2);
        if ((first(1,1)-firstBlock(1,1))^2)+((first(1,2)-firstBlock(1,2))^2)>((first(1,1)-secondBlock(1,1))^2)+((first(1,2)-secondBlock(1,2))^2)
            pointList((2*i)-1,3)=secondBlock(1,1);
            pointList((2*i)-1,4)=secondBlock(1,2);
            pointList(2*i,3)=firstBlock(1,1);
            pointList(2*i,4)=firstBlock(1,2);
        else
            pointList((2*i)-1,3)=firstBlock(1,1);
            pointList((2*i)-1,4)=firstBlock(1,2);
            pointList(2*i,3)=secondBlock(1,1);
            pointList(2*i,4)=secondBlock(1,2);
        end
 pointList(9,1)=xCenter;
pointList(9,2)=yCenter;
pointList(9,3)=xCenter;
pointList(9,4)=yCenter;
%DISPLAYING POINTS FOR ONE LINE---------------------
%     subplot(2,1,1);
%     temp=ones(numRows,numCols)*255;
%     temp(firstBlock(1,1),firstBlock(1,2))=0;
%     temp(secondBlock(1,1),secondBlock(1,2))=0;
%     temp(int16(xCenter),int16(yCenter))=0;
%     temp=imbinarize(temp);
%     temp=imcomplement(temp);
%     imshow(imoverlay(textBlockImage,temp,'cyan'),'InitialMagnification',67);title(['ANGLE : ',num2str(angleList(i,1))]);
%     w = waitforbuttonpress;
    end

    Pimage=ones(numRows,numCols)*255;
    Pimage2=ones(numRows,numCols)*255;
    for k=1:8
        Pimage(pointList(k,1),pointList(k,2))=0;
        Pimage2(pointList(k,3),pointList(k,4))=0;
    end
    Pimage=imbinarize(Pimage);
    Pimage=imcomplement(Pimage);
    
    Pimage2=imbinarize(Pimage2);
    Pimage2=imcomplement(Pimage2);
% subplot(2,1,1);
% imshow(imoverlay(patchI,Pimage,'red'));title(['Text',string]);
% w = waitforbuttonpress;

% imshow(imoverlay(patchI,Pimage2,'red'),'InitialMagnification',67);title(['Text',string]);
% w = waitforbuttonpress;
%fprintf("number of points are %d",size(boundaryList,1))
    cMat=get_transform(pointList(:,1:2),pointList(:,3:4));
%Image Warping and BILINEAR INTERPOLATION----------------------------------------------------------
%------------------------------------------------------------------------
    for i=1:length(pix)
                x=mod(pix(i),numRows);
                if(x==0)
                    x=numRows;
                end
                y=ceil(pix(i)/numRows);
                yt=cMat(1)*x + cMat(2)*y + cMat(3)*(x)^2 + cMat(4)*(y)^2 + cMat(5)*(y)*(x) +cMat(6)*y*(x)^2+ cMat(7)*x*(y)^2 + cMat(8)*(x^2)*(y^2)+cMat(9);
                xt=cMat(10)*x + cMat(11)*y + cMat(12)*(x)^2 + cMat(13)*(y)^2 + cMat(14)*(y)*(x) +cMat(15)*y*(x)^2+ cMat(16)*x*(y)^2 + cMat(17)*(x^2)*(y^2)+cMat(18);
                %fprintf("x,y= %d %d   ->>> xt,yt= %d %d\n ",x,y,xt,yt);
                xt(xt < 1) = 1;
                xt(xt > numCols - 0.001) = numCols - 0.001;
                x1 = floor(xt);
                x2 = x1 + 1;

                yt(yt < 1) = 1;
                yt(yt > numRows - 0.001) = numRows - 0.001;
                y1 = floor(yt);
                y2 = y1 + 1;

      %// 4 Neighboring Pixels
                NP1 = textImage(y1,x1);
                NP2 = textImage(y1,x2);
                NP3 = textImage(y2,x1); 
                NP4 = textImage(y2,x2);

      %// 4 Pixels Weights
                PW1 = (y2-yt)*(x2-xt);
                PW2 = (y2-yt)*(xt-x1);
                PW3 = (x2-xt)*(yt-y1);
                PW4 = (yt-y1)*(xt-x1);

                wrapImage(x,y) = round(PW1 * NP1 + PW2 * NP2 + PW3 * NP3 + PW4 * NP4);
%         wrapImage(x,y)=textImage(round(xt),round(yt));
    end
    for i=1:length(pix)
                x=mod(pix(i),numRows);
                if(x==0)
                    x=numRows;
                end
                y=ceil(pix(i)/numRows);
                if(wrapImage(x,y)==0)
                    wrapImage(x,y)=200;
                end
    end    
%subplot(3,2,6);
imshow(wrapImage);title("wrapImage");
% w = waitforbuttonpress;
%--------------------------------------------------------------------------
%     se = strel('disk', 3);
%     p=imdilate(p,se);
    
end
for labelVal=relevantPatches+1:N
    pix=idx{patches(labelVal,1)};
    for i=1:length(pix)
            x=mod(pix(i),numRows);
            if(x==0)
                x=numRows;
            end
            y=ceil(pix(i)/numRows);
            wrapImage(x,y)=Ibn(x,y);
    end
end
wrapImage=mat2gray(imbinarize(wrapImage));
imshow(wrapImage);title("PICWORDS IMAGE");
%%POST PROCESSING------------------------------------------------------
%----------------------------------------------------------------------
postProcess=ones(numRows,numCols,'uint8')*255;
for i=1:numRows
    for j=1:numCols
        if(wrapImage(i,j)==0)
            postProcess(i,j)=wrapImage(i,j);
        end
    end
end
for labelVal=1:relevantPatches
    pix=idx{patches(labelVal,1)};
    for i=1:length(pix)
            x=mod(pix(i),numRows);
            if(x==0)
                x=numRows;
            end
            y=ceil(pix(i)/numRows);
            if(postProcess(x,y)==255)
                postProcess(x,y)=200;
            end
    end
end


imshow(postProcess);title("PICWORDS IMAGE AFTER POST PROCESSING");
imwrite(postProcess,'output.jpg')
% subplot(3,2,4);imshow(I);title("Color relevant patches");
% subplot(3,2,5);imshow(Igr);title("Grayscale relevant patches");