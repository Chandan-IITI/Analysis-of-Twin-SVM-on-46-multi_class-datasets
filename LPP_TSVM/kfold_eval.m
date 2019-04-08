
function [mean_acc] = seperate_eval(name)

addpath('.\lppTSVM');
%%% datasets can be downloaded from "http://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr/data.tar.gz"
datapath = 'provide the path of your dataset';

tot_data = load([datapath name '\' name '_R.dat']);
index_tune = importdata ([datapath name '\conxuntos.dat']);% for datasets where training-testing partition is available, paramter tuning is based on this file.

%%% Checking whether any index valu is zero or not if zero then increase all index by 1
if length(find(index_tune == 0))>0
    index_tune = index_tune + 1;
end

%%% Remove NaN and store in cell
 for k=1:size(index_tune,1)
  index_sep{k}=index_tune(k,~isnan(index_tune(k,:)));
 end

 %%% Removing first i.e. indexing column and seperate data and classes
data=tot_data(:,2:end);
dataX=data(:,1:end-1);
dataY=data(:,end);
dataYY = dataY; %%% Just replica for further modifying the class label

%%%%%% Normalization start
% do normalization for each feature
mean_X=mean(dataX,1);
dataX=dataX-repmat(mean_X,size(dataX,1),1);
norm_X=sum(dataX.^2,1);
norm_X=sqrt(norm_X);
norm_eval = norm_X; %%% Just save fornormalizing the evaluation data
norm_X=repmat(norm_X,size(dataX,1),1);
dataX=dataX./norm_X;
%%%%%% End of Normalization

%%% Save data in temp folder to reuse
unique_classes = unique(dataYY);

% c1 = [10^-5]; c3 = [10^-5]; c5 = [2^-10]; %%% Just use to check

c1 = [10^-5,10^-4,10^-3,10^-2,10^-1,10^0,10^1,10^2,10^3,10^4,10^5];
c2 = [10^-5,10^-4,10^-3,10^-2,10^-1,10^0,10^1,10^2,10^3,10^4,10^5];

% c5 = scale_range_rbf(dataX);
c5 = [2^-10,2^-9,2^-8,2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3,2^4,2^5,2^6,2^7,2^8,2^9,2^10];

MAX_acc = 0; Resultall = []; count = 0;

for i=1:length(c1)
    %     for j=1:length(c2)
        for m=1:length(c5)
             count = count +1;  %%%% Just displaying the number of iteration    
             
            %%%% Modifying the class label as per TBSVM and chcking whether binaryvclass data or not
all_distance=[];
if (numel(unique_classes)>2)
    for mc=1:numel(unique_classes)
        
        [count mc]
        
        dataY = dataYY;
        dataY(dataYY==unique_classes(mc),:)=1;
        dataY(dataYY~=unique_classes(mc),:)=-1;
        
        %%% Seperation of data
%%% To Tune
trainX=dataX(index_sep{1},:); 
trainY=dataY(index_sep{1},:);
testX=dataX(index_sep{2},:);
testY=dataY(index_sep{2},:);

%%% If dataset needs in TWSVM/TBSVM format
% DataTrain.A = trainX(trainY==1,:);
% DataTrain.B = trainX(trainY==-1,:);

DataTrain = [trainX trainY];
test = [testX testY];                   

                    c=c1(i);

                    kern_para = c5(m);

        [~,~,distan] =lpp_TSVM(DataTrain,test,kern_para,c);
        all_distance=[all_distance,distan];
 end 
else
    error('Data belongs to binary-class, please provide multi-class data');
end

%%% Original labels uses for comparing from predicting value
trainY_orig=dataYY(index_sep{1},:);
testY_orig=dataYY(index_sep{2},:);

                    [~,Predict_Y]=min(all_distance,[],2);

                    if min(unique_classes)== 0 && max(unique_classes)== numel(unique_classes)-1
                        Predict_Y = Predict_Y - 1;
                    else
                        keyboard
                    end
                                               
                    test_accuracy=length(find(Predict_Y==testY_orig))/numel(testY_orig);
                    
                    %%%% Save only optimal parameter with testing accuracy
                    if test_accuracy>MAX_acc % paramater tuning: we prefer the parameter which lead to better accuracy on the test data.
                        MAX_acc=test_accuracy;
                        OptPara.c=c;
                        OptPara.kernPara = kern_para;
                        OptPara.kerntype = 'rbf';
                    end
                    
                    clear Predict_Y;
                end
                %             end
            end


%%%% Training and evaluation with optimal parameter value
clear DataTrain trainX trainY testX testY test;

%%%for datasets where training-testing partition is not available, performance vealuation is based on cross-validation.
 fold_index = importdata([datapath name '\conxuntos_kfold.dat']);
%%% Checking whether any index value is zero or not if zero then increase all index by 1
if length(find(fold_index == 0))>0
    fold_index = fold_index + 1;
end

 for k=1:size(fold_index,1)
  index{k,1}=fold_index(k,~isnan(fold_index(k,:)));
 end

 for f=1:4
     all_distance=[];
     for mc=1:numel(unique_classes)
         
        dataY = dataYY;      
        dataY(dataYY==unique_classes(mc),:)=1;
        dataY(dataYY~=unique_classes(mc),:)=-1;
        
        trainX=dataX(index{2*f-1},:);
        trainY=dataY(index{2*f-1},:);
        testX=dataX(index{2*f},:);
        testY=dataY(index{2*f},:);
     
%      DataTrain.A = trainX(trainY==1,:);
%      DataTrain.B = trainX(trainY==-1,:);
     
DataTrain = [trainX trainY];
test = [testX testY];

     [~,~,distan] =lpp_TSVM(DataTrain,test,OptPara.kernPara,OptPara.c);
     all_distance=[all_distance,distan];
     end
     
%%% Original labels uses for comparing from predicting value
trainY_orig=dataYY(index{2*f-1},:);
testY_orig=dataYY(index{2*f},:);

     [~,Predict_Y]=min(all_distance,[],2);

      if min(unique_classes)== 0 && max(unique_classes)== numel(unique_classes)-1
         Predict_Y = Predict_Y - 1;
      else
         keyboard
      end
                    
     test_acc(f)=length(find(Predict_Y==testY_orig))/numel(testY_orig);
     
     clear Predict_Y DataTrain trainX trainY testX testY;
 end
 
 mean_acc = mean(test_acc)
 OptPara.test_acc = mean_acc*100;
 
filename = ['Res_' name '.mat'];
save (filename, 'OptPara');

rmpath('.\lppTSVM');
end
