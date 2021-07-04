function [ Sw,Sb,St ] = scatter_mat ( X,y )
%Function     --ʵ�����ڡ���䡢���ɢ������ �������
%X            --���๹�ɵ��������ϣ�һ�������� ��ʾ һ�������� 
%y            --һ��Nά����������i��Ԫ�ذ���X�е�i��������label���ܹ���c����꣩
%Sw           --����ɢ���������ھ��� �� ƽ����ʽ
%Sb           --���ɢ������������ �� ƽ����ʽ
%Sw           --���ɢ������
% load self_GK_data;
% Blabel=ones(1,403);
% Blabel(201:403)=2;
% X=self_GK_data';
% y=Blabel;

[L,N]=size(X);  %��X��L*Nά
c=max(y);
 
% Sw
m=[];
Sw=zeros(1);
for i=1:1:c
    y_temp=(y==i);
    X_temp=X(:,y_temp);
    P(i)=sum(y_temp)/(N*L);
    m(:,i)=(mean(X_temp'))';
    Sw=Sw+P(i)*cov(X_temp');  %������ʽ
end
 
% Sb
m0=(sum(((ones(L,1)*P).*m)'))';
Sb=zeros(1);
for i=1:c
    Sb=Sb+P(i)*((m(:,i)-m0)*(m(:,i)-m0)');  %������ʽ
end
 
% St
St=Sw+Sb; %������ʽ

