%load maxlittle_plearn_pratearray
%load tmparray(PartB)
%load testresultdata4_30
load testresultdata_polyp
lrate=[];
core_index=zeros(1,8);
for core_num=2:8
prate=zeros(core_num,10);   
for nexp=1:10
for dim=1:core_num     
 prate(dim,nexp)=max(testresultdata(core_num,dim,nexp,:));
% end
%[~,dimindex]=max(dimarray);
%prate(dim,nexp)=tmprate(core_num,dim,nexp);
end
end
pratevec=sum(prate);
[~,core_index(core_num)]=max(pratevec);
lrate=[lrate;prate];
%nrate(:,core_num)=sum(prate,2)/10;
end

for i=1:35   
   [~,index(i)]=max(lrate(i,:));
end
% frate=zeros(35,1);
% for i=1:35   
% frate(i)=lrate(i,index(i));
% end
save('seqindex_polyp', 'core_index');