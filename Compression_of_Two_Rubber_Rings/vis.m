clc,clear,close all
%%
ps=3;
E1=1e2;E2=1e2;
mu=0.3;
la1 = mu / (1 + mu) / (1 - 2 * mu) * E1;
nu1 = 1 / (1 + mu) / 2 * E1;
la2 = mu / (1 + mu) / (1 - 2 * mu) * E2;
nu2 = 1 / (1 + mu) / 2 * E2;

Loss=[];
kk=0;
T=0;
for j=1:12
    for i=20

        load(['out_ls',num2str(j),'_tl',num2str(i-1),'.mat'])
        [sig11_1, sig12_1, sig22_1]=cal_stress(F11_1,F12_1,F21_1,F22_1,nu1,la1);
        [sig11_2, sig12_2, sig22_2]=cal_stress(F11_2,F12_2,F21_2,F22_2,nu2,la2);

        vs_1=sqrt(sig11_1.^2+sig22_1.^2-sig11_1.*sig22_1+3*sig12_1.^2);
        vs_2=sqrt(sig11_2.^2+sig22_2.^2-sig11_2.*sig22_2+3*sig12_2.^2);

        figure(1)

        scatter((x2(:,1)+u2+0.9),x2(:,2)+v2+0.8,ps,vs_1,'filled'),hold on
        % scatter(100,100)
        scatter((x1(:,1)+u1+0.35),x1(:,2)+v1+0.35,ps,vs_2,'filled')
        % scatter(100,100)

        plot([0 0],[1.5 0],'k','LineWidth',1.),hold on
        plot([0 1.25],[0 0],'k','LineWidth',1.)
        plot([1.25 1.25],[1.5 0],'k','LineWidth',1.)

        plot([0 1.25],[1 1]*(1.15-0.05*j),'k','LineWidth',1.)

        title(['PINN Loading Step: ',num2str(j)])
        colorbar
        colormap jet
        axis equal
        hold off
        axis([-0.15 1.4 -0.15 1.4])
        box on
        xlabel('x (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
        ylabel('y (m)','Fontname', 'Helvetica','FontWeight','bold','FontSize',9)
        set(gcf,'position',[0,100,400,400])

    end
end
