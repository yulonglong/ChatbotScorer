clear
chatbotname={'IRIS','Joker','TickTock'};

%%%% Chatbot Loop Starts
for nchatbot=1:3
chatbot=chatbotname{nchatbot};
hf=figure(); set(hf,'Name',chatbot,'Color',[1 1 1]);

%%%% Fold Loop Starts
for fold=0:9
folder=[chatbot,'_dump/'];
filename=sprintf('%s_fold%d_',chatbot,fold);

% Reads the train set and computes the distributions of labels 
fid=fopen([folder,filename,'train.txt'],'rt');
contents = fscanf(fid,'%c',Inf);
fclose(fid);
n_inval = max(size(strfind(contents,'INVALID')));
n_valid = max(size(strfind(contents,'VALID'))) - n_inval;
n_accep = max(size(strfind(contents,'ACCEPTABLE')));
model = cumsum([n_inval, n_accep, n_valid])/sum([n_inval, n_accep, n_valid]);

% Reads the test set and computes the scores 
fid=fopen([folder,filename,'test.txt'],'rt');
theline = fgetl(fid);
index=0;
while 1
    theline = fgetl(fid);
    if theline==-1, break; end
    if not(isempty(strtrim(theline)))
        n_inval = max(size(strfind(theline,'INVALID')));
        n_valid = max(size(strfind(theline,'VALID'))) - n_inval;
        n_accep = max(size(strfind(theline,'ACCEPTABLE')));
        index=index+1;
        scores(index)= (0*n_inval+0.5*n_accep+1*n_valid)/(n_inval+n_accep+n_valid);
    end
end
fclose(fid);

% Computes 1K random scores following the train distributions and computes
% the correlations with the test scores (plots the histogram and gets the 
% mean values and standard deviations)
nsimulations=1000;
baselines = rand(length(scores),nsimulations);
baselines(baselines<model(1))=0;
baselines((baselines>0)&(baselines<model(2)))=0.5;
baselines(baselines>0.5)=1;

for k=1:nsimulations
    [cc,pv]=corrcoef(scores',baselines(:,k));
    pearsoncc(k)=cc(1,2);
    pvalue(k)=pv(1,2);
end
subplot(2,5,fold+1); hist(pearsoncc,20); title(sprintf('fold-%d',fold));
pearsoncc_mean = mean(pearsoncc);
pearsoncc_std = std(pearsoncc);

% Saves results in the report
fid=fopen('report.txt','at');
fprintf(fid,'%s\tfold-%d\tpearson: mean = %+6.4f\tstd = %6.4f \n',chatbot(1:min(5,length(chatbot))),fold,pearsoncc_mean,pearsoncc_std);
fclose(fid);

clear scores baselines pearsoncc pvalue pearsoncc_mean pearsoncc_std

end %%%%% Fold Loop Ends
    
end %%%%% Chatbot Loop Ends



