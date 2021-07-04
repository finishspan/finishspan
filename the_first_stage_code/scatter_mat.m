function [ Sw,Sb,St ] = scatter_mat ( X,y )
%Function     --实现类内、类间、混合散布矩阵 距离计算
%X            --多类构成的样本集合（一个列向量 表示 一个样本） 
%y            --一个N维行向量，第i个元素包含X中第i个向量的label（总共有c个类标）
%Sw           --类内散布矩阵，类内距离 的 平方形式
%Sb           --类间散布矩阵，类间距离 的 平方形式
%Sw           --混合散布矩阵
% load self_GK_data;
% Blabel=ones(1,403);
% Blabel(201:403)=2;
% X=self_GK_data';
% y=Blabel;

[L,N]=size(X);  %设X有L*N维
c=max(y);
 
% Sw
m=[];
Sw=zeros(1);
for i=1:1:c
    y_temp=(y==i);
    X_temp=X(:,y_temp);
    P(i)=sum(y_temp)/(N*L);
    m(:,i)=(mean(X_temp'))';
    Sw=Sw+P(i)*cov(X_temp');  %矩阵形式
end
 
% Sb
m0=(sum(((ones(L,1)*P).*m)'))';
Sb=zeros(1);
for i=1:c
    Sb=Sb+P(i)*((m(:,i)-m0)*(m(:,i)-m0)');  %矩阵形式
end
 
% St
St=Sw+Sb; %矩阵形式

