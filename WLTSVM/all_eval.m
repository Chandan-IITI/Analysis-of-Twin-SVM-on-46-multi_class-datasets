
clear
clc
warning('off','all')

names = importdata('names.txt');

for datanum = 1:length(names)
    datanum
          if(~ismember(datanum, [1 5 7 20 38 52 68 73 75 77 85 86 87 88 92 101 102 114 115 116 119]))   %%% 46 Datasets remain after removing these time cosuming datasets 

name = names{datanum};
acc = 0;
try
acc = separate_eval(name);
catch
    try
   acc = kfold_eval(name); 
    catch
        disp('data belongs to binary-class classification or dataset not found')
        continue
    end
end
  
if acc ~= 0
    xlRange1 = ['A' num2str(datanum)];
    xlswrite('all_results.xlsx', {name}, 1, xlRange1);
    xlRange2 = ['B' num2str(datanum)];
    xlswrite('all_results.xlsx', acc*100, 1, xlRange2);
end
    
end

end