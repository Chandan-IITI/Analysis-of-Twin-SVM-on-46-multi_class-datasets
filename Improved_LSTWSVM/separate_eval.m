
function [test_acc] = seperate_eval(name)

addpath('.\improved_LSTWSVM');
%%% datasets can be downloaded from "http://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr/data.tar.gz"
datapath = 'provide the path of your dataset';

train = load ([datapath name '\' name '_train_R.dat']);% for datasets where training-testing partition is available, paramter tuning is based on this file.
index_tune = importdata ([datapath name '\conxuntos.dat']);% for datasets where training-testing partition is available, paramter tuning is based on this file.
test_eval = load ([datapath name '\' name '_test_R.dat']);% for datasets where training-testing partition is available, paramter tuning is based on this file.

%%% Checking whether any index valu is zero or not if zero then increase all index by 1
if length(find(index_tune == 0))>0
    index_tune = index_tune + 1;
end

%%% Remove NaN and store in cell
 for k=1:size(index_tune,1)
  index_sep{k}=index_tune(k,~isnan(index_tune(k,:)));
 end

%%% To Evaluate
test_data = test_eval(:,2:end-1);
test_label = test_eval(:,end);
test_lab = test_label;  %%% Just replica for further modifying the class label

%%% To Tune
dataX=train(:,2:end-1);
dataY=train(:,end);
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

%%%% Normalize the evaluation data
norm_ev = repmat(norm_eval,size(test_data,1),1);
test_data=test_data./norm_ev;
%%%% End of normalization of evaluation data
%%%% End of Normalization %%%%

%%% Save data in temp folder to reuse
unique_classes = unique(dataYY);

% c1 = [10^-5]; c3 = [10^-5]; c5 = [2^-10]; %%% Just use to check

c1 = [10^-5,10^-4,10^-3,10^-2,10^-1,10^0,10^1,10^2,10^3,10^4,10^5];
c2 = [10^-5,10^-4,10^-3,10^-2,10^-1,10^0,10^1,10^2,10^3,10^4,10^5];
c3 = [10^-5,10^-4,10^-3,10^-2,10^-1,10^0,10^1,10^2,10^3,10^4,10^5]; %%% Eps1
c4 = [10^-5,10^-4,10^-3,10^-2,10^-1,10^0,10^1,10^2,10^3,10^4,10^5]; %%% Eps2
% c5 = scale_range_rbf(dataX);
c5 = [2^-10,2^-9,2^-8,2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3,2^4,2^5,2^6,2^7,2^8,2^9,2^10];

MAX_acc = 0; Resultall = []; count = 0;

for i=1:length(c1)
    %     for j=1:length(c2)
    for k=1:length(c3)
        %             for l=1:length(c4)
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

                    FunPara.c1=c1(i);
                    %                     FunPara.c2=c2(j);
                    FunPara.c2=c1(i);
                    
                    FunPara.c3=c3(k);
                    %                     FunPara.c4=c4(l);
                    FunPara.c4=c3(k);
                    
                    FunPara.kerfPara.type = 'rbf';
                    FunPara.kerfPara.pars = c5(m);
        
        [~,~,~,~,~,~,distan] =Improved_LSTWSVM(test,DataTrain,FunPara);
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
                        OptPara.c1=FunPara.c1; OptPara.c2=FunPara.c2; OptPara.c3=FunPara.c3; OptPara.c4=FunPara.c4;
                        OptPara.kerfPara.type = FunPara.kerfPara.type; OptPara.kerfPara.pars = FunPara.kerfPara.pars;
                    end
                    
                    clear Predict_Y;
                end
                             end
            end


%%%% Training and evaluation with optimal parameter value
clear DataTrain trainX trainY testX testY test;

% DataTrain.A = dataX(dataY==1,:);
% DataTrain.B = dataX(dataY==-1,:);
     all_distance=[];
    for mc=1:numel(unique_classes)
        
        dataY = dataYY;
        dataY(dataYY==unique_classes(mc),:)=1;
        dataY(dataYY~=unique_classes(mc),:)=-1;
        
        %%% For valuation on test data
        test_label = test_lab;
        test_label(test_lab==unique_classes(mc),:)=1;
        test_label(test_lab~=unique_classes(mc),:)=-1;
        
        DataTrain = [dataX dataY];
        test = [test_data test_label];
        
        [~,~,~,~,~,~,distan] =Improved_LSTWSVM(test,DataTrain,OptPara);
        all_distance=[all_distance,distan];
    end

      [~,Predict_Y]=min(all_distance,[],2);

      if min(unique_classes)== 0 && max(unique_classes)== numel(unique_classes)-1
         Predict_Y = Predict_Y - 1;
      else
         keyboard
      end
      
test_acc = length(find(Predict_Y==test_lab))/numel(test_lab)
OptPara.test_acc = test_acc*100;

filename = ['Res_' name '.mat'];
save (filename, 'OptPara');

rmpath('.\improved_LSTWSVM');
end
