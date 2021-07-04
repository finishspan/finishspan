function [ data_temp ] = distant(r1, r2, data, x, lei, k)
%=====����relief�㷨�ĵ�����function [w , m] = weight( r1, r2, data )�е�һ���Ӻ�����
%=====�������ܣ��������x�ҵ�һ��m��2k����
%=====����������ÿ�����������ں������롣�˴�Ϊ��ŷ�Ͼ��롣���ɸ�����Ҫѡ��ͬ�ķ�����
% =====r1��r2�ֱ������������ĸ�����
% =====data�Ǳ�׼��֮��Ļ�����������ݡ�
%=====x������������
%=====lei�����lei��1��ʾ�������ڵ�һ�ࣻlei��0��ʾ�������ڵڶ��ࡣ
%=====k��r1��r2�е�Сֵ��
%=====data_temp��2k�������Ļ������������СΪ������������2k��������ǰ���k������Ϊ���������������k��Ϊ���������
%=====data_temp������ѡ���ԭ�򣺰�������x������ͬ������֮��ľ����С����ѡ����k��ͬ����������������x����������֮�䰴�վ���Ӵ�Сѡ����k������������
[m, n] = size( data );
%���ȣ�����ÿ���������������ھ��롣
dist_temp = zeros(1, n);
for i = 1: n       %����ѭ��
    y = data(:, i);
    dist_temp(i) = sqrt(sum((x-y).*(x-y)));  %�Ƿ���ŷ�Ͼ���ĺ�����
end
if lei == 1 %��ʾ�������ڵ�һ��
    [b1, ix1] = sort (dist_temp(1: r1));  %���ڣ���С��������ѡ�����С��k����
    %[b2, ix2] = sort (dist_temp(r1+1: n), 'descend');%��䣺�Ӵ�С����ѡ�������k����
    [b2, ix2] = sort (dist_temp(r1+1: n));%��䣺�Ӵ�С����ѡ�������k����
    data_temp(:, 1: k) = data (:,ix1(1: k));  %ix1(b-k+1:b)
    data_temp(:, k+1:2*k) = data (:,(r1+ix2(1: k)));
else      %��ʾ�������ڵڶ��ࡣ
    [b1,ix1] = sort (dist_temp(r1+1:n));%���ڣ���С��������ѡ�����С��k����
    %[b2,ix2] = sort (dist_temp(1:r1), 'descend' );  %��䣺�Ӵ�С����ѡ�������k����
    [b2,ix2] = sort (dist_temp(1:r1));  %��䣺�Ӵ�С����ѡ�������k����
    data_temp(:,1:k) = data (:, (r1+ix1(1:k)));  %ix1(b-k+1:b)
    data_temp(:,k+1:2*k) = data (:, ix2(1:k));
end
%end of function [data_hm]=distant(r1,r2,data,x,lei,k)