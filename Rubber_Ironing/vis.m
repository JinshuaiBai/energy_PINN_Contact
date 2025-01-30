clc,clear,close all
%%
k=0;
ps=2;
E1=3e2;E2=1e2;
mu=0.3;
la1 = mu / (1 + mu) / (1 - 2 * mu) * E1;
nu1 = 1 / (1 + mu) / 2 * E1;
la2 = mu / (1 + mu) / (1 - 2 * mu) * E2;
nu2 = 1 / (1 + mu) / 2 * E2;

for i = 0:30
    k=k+1;
    load(['out_vis_',num2str(i),'.mat'])

    [sig11_1, sig12_1, sig22_1]=cal_stress(F11_1,F12_1,F21_1,F22_1,nu1,la1);
    [sig11_2, sig12_2, sig22_2]=cal_stress(F11_2,F12_2,F21_2,F22_2,nu2,la2);

    figure(1)
    scatter(x1(:,1)+u1,x1(:,2)+2+v1,ps,sig22_1,'filled'),hold on
    scatter(x2(:,1)+u2,x2(:,2)+1.5+v2,ps,sig22_2,'filled')
    colorbar
    colormap jet
    axis equal
    box on
    axis([-3 3 -0.1 3.1])
    xlabel('x (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
    ylabel('y (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
    title(['\sigma_y Loading Step: ',num2str(i)])
    set(gcf,'position',[0,500,320,200])

    drawnow
    hold off

end

%%
