
testresultdata=zeros(40,40,10,337);
testresultdata1=zeros(40,40,10,337);
testresultdata2=zeros(40,40,10,337);
testStarry=zeros(40,40,10,337);
for core_num=2:8
% plearn_trainkernels(core_num);

for dim=1:core_num
for nexp=1:10
[srate,srate1,srate2,cftcun]=plearn_featureextraction_relief_xinan(core_num,nexp,dim);
testresultdata(core_num,dim,nexp,:)=srate;
% testresultdata1(core_num,dim,nexp,:)=srate1;
% testresultdata2(core_num,dim,nexp,:)=srate2;
% [avalue,bindex]=max(srate);
% avalue1=srate1(bindex);
% avalue2=srate2(bindex);
% testStarry(core_num,dim,nexp,:)=Starry;
% save testresultdata;
end
end

end

save('testresultdata', 'testresultdata');