
%==========================================================================
%****************** ****3D拓扑优化初始模板（带中文注释）********************
%*********************************禁止修改*********************************
%****************************date：2023.03.14******************************
%==========================================================================

function [x] = top3d(nelx,nely,nelz,volfrac,penal,rmin)
nelx = 100;
nely = 100;
nelz = 100;
volfrac = 0.15;
penal = 3;
rmin =1.5;
%%
%用户自定义迭代参数                            
maxloop = 200;                                                                  % 最大迭代次数                             
tolx = 0.01;                                                                    % 终止规则
displayflag = 1;                                                                % 图像显示标志
%%
% 用户自定义材料属性
E0 = 1;                                                                         % 固体材料的杨氏模量
Emin = 1e-9;                                                                    % 空隙材料的杨氏模量
nu = 0.3;                                                                       % 泊松比
%%
% 用户自定义加载力的自由度
[il,jl,kl] = meshgrid(nelx, 0, 0:nelz);                                         % 外力所作用的节点的坐标列表
loadnid = kl*(nelx+1)*(nely+1)+il*(nely+1)+(nely+1-jl);                         % 外力所作用的节点的顺序标号
loaddof = 3*loadnid(:) - 2;                                                     % 外力所作用的自由度的顺序标号
%%
% 用户自定义固定的自由度
[iif,jf,kf] = meshgrid(0,0:nely,0:nelz);                                        % 需固定的节点的坐标列表
fixednid = kf*(nelx+1)*(nely+1)+iif*(nely+1)+(nely+1-jf);                       % 需固定的节点的顺序标号
fixeddof = [3*fixednid(:); 3*fixednid(:)-1; 3*fixednid(:)-2];                   % 需固定节点的对应自由度的顺序标号
%%
% 有限元分析参数预定义
nele = nelx*nely*nelz;
ndof = 3*(nelx+1)*(nely+1)*(nelz+1);
F = sparse(loaddof,1,-1 ,ndof,1);                                               %初始化在设计区域作用的力
U = zeros(ndof,1);                                                              %初始化由载荷力引起的节点位移
freedofs = setdiff(1:ndof,fixeddof);                                            %初始化活动自由度
%%
%生成全局刚度矩阵
KE = lk_H8(nu);                                                                 %初始化单元刚度矩阵
nodegrd = reshape(1:(nely+1)*(nelx+1),nely+1,nelx+1);                           %生成x-y(z=0)平面的单元标号
nodeids = reshape(nodegrd(1:end-1,1:end-1),nely*nelx,1);            
nodeidz = 0:(nely+1)*(nelx+1):(nelz-1)*(nely+1)*(nelx+1);
nodeids = repmat(nodeids,size(nodeidz))+repmat(nodeidz,size(nodeids));
edofVec = 3*nodeids(:)+1;
edofMat = repmat(edofVec,1,24)+ ...
    repmat([0 1 2 3*nely + [3 4 5 0 1 2] -3 -2 -1 ...
3*(nely+1)*(nelx+1)+[0 1 2 3*nely + [3 4 5 0 1 2] -3 -2 -1]],nele,1);           %生成每个单元的所有节点自由度的标号
iK = reshape(kron(edofMat,ones(24,1))',24*24*nele,1);
jK = reshape(kron(edofMat,ones(1,24))',24*24*nele,1);


%{
=============================================================================
Tips：
1.F中一列的长度等于设计区域中自由度数量的总数，每一列对应加载在设计区域中的一个力
2.U中一列的长度等于设计区域中自由度数量的总数，每一列对应F中对应列力引起的节点位移
3.edofMat中一行对应一个单元8个节点的24个自由度的标号
=============================================================================
%}
%%
% 准备过滤器
iH = ones(nele*(2*(ceil(rmin)-1)+1)^2,1);
jH = ones(size(iH));
sH = zeros(size(iH));
k = 0;
for k1 = 1:nelz
    for i1 = 1:nelx
        for j1 = 1:nely
            e1 = (k1-1)*nelx*nely + (i1-1)*nely+j1;
            for k2 = max(k1-(ceil(rmin)-1),1):min(k1+(ceil(rmin)-1),nelz)
                for i2 = max(i1-(ceil(rmin)-1),1):min(i1+(ceil(rmin)-1),nelx)
                    for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),nely)
                        e2 = (k2-1)*nelx*nely + (i2-1)*nely+j2;
                        k = k+1;
                        iH(k) = e1;
                        jH(k) = e2;
                        sH(k) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2+(k1-k2)^2));
                    end
                end
            end
        end
    end
end
H = sparse(iH,jH,sH);
Hs = sum(H,2);

%%
% 初始化迭代参数
x = repmat(volfrac,[nely,nelx,nelz]);                                           %初始化密度矩阵
xPhys = x; 
loop = 0;                                                                       %迭代次数
change = 1;                                                                     %优化目标值
%{
=================================================================================
Tips：
1.每个单元的初始密度值设置为体积分数volfrac
2.优化迭代设置两个终止条件：1）迭代次数loop > maxloop 2）优化目标值change < tolx
=================================================================================
%}
%%
% 开始迭代
while change > tolx && loop < maxloop
    loop = loop+1;

    % 有限元分析
    sK = reshape(KE(:)*(Emin+xPhys(:)'.^penal*(E0-Emin)),24*24*nele,1);
K = sparse(iK,jK,sK); K = (K+K')/2;                                             %生成全局刚度矩阵
    U(freedofs,:) = K(freedofs,freedofs)\F(freedofs,:);                         %计算活动自由度在加载力作用下的位移
    
    % 计算目标函数值以及敏感度分析（Optimality Criteria OC 最优化准则算法）
    ce = reshape(sum((U(edofMat)*KE).*U(edofMat),2),[nely,nelx,nelz]);          %计算每个单元的最小柔度的目标值FU，并于单元标号对应
    c = sum(sum(sum((Emin+xPhys.^penal*(E0-Emin)).*ce)));                       %计算总体的最小柔度目标值并求和
    dc = -penal*(E0-Emin)*xPhys.^(penal-1).*ce;
    dv = ones(nely,nelx,nelz);
    %{
    %======================================================================
    Tips：
    1.U(edofMat)是将U中的位移值按照edofMat中每个节点的排列顺序进行排列，
      即把对应节点的位移按顺序放到对应的一行
    2.敏感度可以理解为梯度
     ======================================================================
    %}
    
    % 敏感度的过滤和修改
    dc(:) = H*(dc(:)./Hs);  
    dv(:) = H*(dv(:)./Hs);
    % 根据优化准则对设计参数x(密度)进行更新
    l1 = 0; l2 = 1e9; move = 0.2;
    while (l2-l1)/(l1+l2) > 1e-3
        lmid = 0.5*(l2+l1);
        xnew = max(0,max(x-move,min(1,min(x+move,x.*sqrt(-dc./dv/lmid)))));
        xPhys(:) = (H*xnew(:))./Hs;
        if sum(xPhys(:)) > volfrac*nele, l1 = lmid; else l2 = lmid; end
    end
    change = max(abs(xnew(:)-x(:)));
    x = xnew;
    % 打印结果
    fprintf(' It.:%5i Obj.:%11.4f Vol.:%7.3f ch.:%7.3f\n',loop,c,mean(xPhys(:)),change);
    % 绘制密度体素图
    if displayflag, clf; display_3D(xPhys); end 
end
clf; display_3D(xPhys);
end


% ============= 生成单元刚度矩阵 =============
function [KE] = lk_H8(nu)
A = [32 6 -8 6 -6 4 3 -6 -10 3 -3 -3 -4 -8;
    -48 0 0 -24 24 0 0 0 12 -12 0 12 12 12];
k = 1/144*A'*[1; nu];

K1 = [k(1) k(2) k(2) k(3) k(5) k(5);
    k(2) k(1) k(2) k(4) k(6) k(7);
    k(2) k(2) k(1) k(4) k(7) k(6);
    k(3) k(4) k(4) k(1) k(8) k(8);
    k(5) k(6) k(7) k(8) k(1) k(2);
    k(5) k(7) k(6) k(8) k(2) k(1)];
K2 = [k(9)  k(8)  k(12) k(6)  k(4)  k(7);
    k(8)  k(9)  k(12) k(5)  k(3)  k(5);
    k(10) k(10) k(13) k(7)  k(4)  k(6);
    k(6)  k(5)  k(11) k(9)  k(2)  k(10);
    k(4)  k(3)  k(5)  k(2)  k(9)  k(12)
    k(11) k(4)  k(6)  k(12) k(10) k(13)];
K3 = [k(6)  k(7)  k(4)  k(9)  k(12) k(8);
    k(7)  k(6)  k(4)  k(10) k(13) k(10);
    k(5)  k(5)  k(3)  k(8)  k(12) k(9);
    k(9)  k(10) k(2)  k(6)  k(11) k(5);
    k(12) k(13) k(10) k(11) k(6)  k(4);
    k(2)  k(12) k(9)  k(4)  k(5)  k(3)];
K4 = [k(14) k(11) k(11) k(13) k(10) k(10);
    k(11) k(14) k(11) k(12) k(9)  k(8);
    k(11) k(11) k(14) k(12) k(8)  k(9);
    k(13) k(12) k(12) k(14) k(7)  k(7);
    k(10) k(9)  k(8)  k(7)  k(14) k(11);
    k(10) k(8)  k(9)  k(7)  k(11) k(14)];
K5 = [k(1) k(2)  k(8)  k(3) k(5)  k(4);
    k(2) k(1)  k(8)  k(4) k(6)  k(11);
    k(8) k(8)  k(1)  k(5) k(11) k(6);
    k(3) k(4)  k(5)  k(1) k(8)  k(2);
    k(5) k(6)  k(11) k(8) k(1)  k(8);
    k(4) k(11) k(6)  k(2) k(8)  k(1)];
K6 = [k(14) k(11) k(7)  k(13) k(10) k(12);
    k(11) k(14) k(7)  k(12) k(9)  k(2);
    k(7)  k(7)  k(14) k(10) k(2)  k(9);
    k(13) k(12) k(10) k(14) k(7)  k(11);
    k(10) k(9)  k(2)  k(7)  k(14) k(7);
    k(12) k(2)  k(9)  k(11) k(7)  k(14)];
KE = 1/((nu+1)*(1-2*nu))*...
    [ K1  K2  K3  K4;
    K2'  K5  K6  K3';
    K3' K6  K5' K2';
    K4  K3  K2  K1'];
end
% === DISPLAY 3D TOPOLOGY (ISO-VIEW) ===
function display_3D(rho)
[nely,nelx,nelz] = size(rho);
hx = 1; hy = 1; hz = 1;            % User-defined unit element size
face = [1 2 3 4; 2 6 7 3; 4 3 7 8; 1 5 8 4; 1 2 6 5; 5 6 7 8];
set(gcf,'Name','ISO display','NumberTitle','off');
for k = 1:nelz
    z = (k-1)*hz;
    for i = 1:nelx
        x = (i-1)*hx;
        for j = 1:nely
            y = nely*hy - (j-1)*hy;
            if (rho(j,i,k) > 0.5)  % User-defined display density threshold
                vert = [x y z; x y-hx z; x+hx y-hx z; x+hx y z; x y z+hx;x y-hx z+hx; x+hx y-hx z+hx;x+hx y z+hx];
                vert(:,[2 3]) = vert(:,[3 2]); vert(:,2,:) = -vert(:,2,:);
                patch('Faces',face,'Vertices',vert,'FaceColor',[0.2+0.8*(1-rho(j,i,k)),0.2+0.8*(1-rho(j,i,k)),0.2+0.8*(1-rho(j,i,k))]);
                hold on;
            end
        end
    end
end
axis equal; axis tight; axis off; box on; view([30,30]); pause(1e-6);
end
% =========================================================================
% === This code was written by K Liu and A Tovar, Dept. of Mechanical   ===
% === Engineering, Indiana University-Purdue University Indianapolis,   ===
% === Indiana, United States of America                                 ===
% === ----------------------------------------------------------------- ===
% === Please send your suggestions and comments to: kailiu@iupui.edu    ===
% === ----------------------------------------------------------------- ===
% === The code is intended for educational purposes, and the details    ===
% === and extensions can be found in the paper:                         ===
% === K. Liu and A. Tovar, "An efficient 3D topology optimization code  ===
% === written in Matlab", Struct Multidisc Optim, 50(6): 1175-1196, 2014, =
% === doi:10.1007/s00158-014-1107-x                                     ===
% === ----------------------------------------------------------------- ===
% === The code as well as an uncorrected version of the paper can be    ===
% === downloaded from the website: http://www.top3dapp.com/             ===
% === ----------------------------------------------------------------- ===
% === Disclaimer:                                                       ===
% === The authors reserves all rights for the program.                  ===
% === The code may be distributed and used for educational purposes.    ===
% === The authors do not guarantee that the code is free from errors, a